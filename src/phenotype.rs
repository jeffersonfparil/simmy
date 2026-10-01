use crate::{genotype::*, linalg::kernel};
use crate::linalg::context::GpuContext;
use crate::linalg::kernel::GpuKernel;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use rand::seq::index;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Trait {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Linear,
    Sigmoid,
    ReLU,
}

#[derive(Debug)]
pub struct GenoPhenoNetwork {
    // We've opt to generalise the genetic model as a multilayer perceptron or artificial neural network with 0 or more hidden layers translating to a vector of 1 or more GpuTensors of weights and biases.
    pub weights: Vec<GpuTensor>,
    pub biases: Vec<GpuTensor>,
    pub activations: Vec<Activation>,
}

impl GpuTensor {
    pub fn activate(&mut self, ctx: &GpuContext, activation: &Activation) -> Result<()> {
        match activation {
            Activation::Linear => (),
            Activation::Sigmoid => {
                let kernel = GpuKernel::new(ctx);
                let denominator = kernel.add_scalar(&kernel.exp(&kernel.neg(self)?)?, 1.0)?;
                self.buffer = kernel.pow_scalar(&denominator, -1.0)?.buffer.clone();
            },
            Activation::ReLU => {
                let kernel = GpuKernel::new(ctx);
                let greater_than_zero = kernel.gt_scalar(self, 0.0)?;
                self.buffer = kernel.mul(self, &greater_than_zero)?.buffer.clone();
            }
        }
        Ok(())
    }
}

impl GenoPhenoNetwork {
    pub fn check(&self) -> Result<()> {
        ensure!(self.weights.len() > 0, "Model is undefined! No weights found!");
        ensure!(self.weights.len() == self.biases.len(), "The weights ({}) and biases ({}) are incompatible!", self.weights.len(), self.biases.len());
        ensure!(self.weights.len() == self.activations.len(), "The weights ({}) and activations ({}) are incompatible!", self.weights.len(), self.activations.len());
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            ensure!(w.shape.len() == 2, "We expect weights to be 2D tensors! The {}th weight has shape: {:?}!", i, w.shape);
            ensure!(b.shape.len() == 2, "We expect biases to be 2D tensors! The {}th bias has shape: {:?}!", i, b.shape);
            ensure!(b.shape[1] == 1, "The second dimension of the {}ith biases should be of length 1!", i);
            ensure!(w.shape[1] == b.shape[0], "The {}ith weights and biases are incompatible!", i);
            if i > 0 {
                ensure!(self.weights[i-1].shape[1] == w.shape[0], "The {}ith and {}ith weights are incompatible!", i-1, i);
            }
        }
        ensure!(self.weights[self.weights.len() - 1].shape[1] >= 1, "We expect at least a single output node per entry! Each node represent a trait!");
        Ok(())
    }
    pub fn predict(&self, ctx: &GpuContext, input: &GpuTensor) -> Result<GpuTensor> {
        self.check()?;
        ensure!(input.shape.len() == 3, "We expect the input to be a 3D tensor (entries x loci-alleles x homologous chromosomes)!");
        ensure!(self.weights[0].shape[0] == input.shape[1], "The model (1st weight shape: {:?}) and input (shape: {:?}) are incompatible!", self.weights[0].shape, input.shape);
        let n_entries: usize = input.shape[0] as usize;
        let n_loci_alleles: usize = input.shape[1] as usize;
        ensure!(input.shape[2] == 2, "We expect the 3rd dimension to have 2 levels, one for each homologous chromosome or one from each parent!");
        let kernel = GpuKernel::new(ctx);
        let mut output: GpuTensor = {
            // Note that we are collapsing the allele dosages of the 2 homologous chromosomes and we are not accounting for haplotype effects/phase information for simplicity!
            let x: GpuTensor = kernel.add(&input.slice_view(&[(0, n_entries), (0, n_loci_alleles), (0, 1)])?, &input.slice_view(&[(0, n_entries), (0, n_loci_alleles), (1, 2)])?)?;
            let xw: GpuTensor = kernel.contract(&x, &self.weights[0])?;
            let mut y = kernel.add(&xw, &self.biases[0])?;
            y.activate(ctx, &self.activations[0])?;
            y
        };
        for ((w, b), a) in self.weights.iter().zip(self.biases.iter()).zip(self.activations.iter()).skip(1) {
            output = kernel.add(&kernel.contract(&output, w)?, b)?;
            output.activate(ctx, a)?;
        }
        Ok(output)
    }
}

#[derive(Debug)]
pub struct GeneticModel {
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub traits: Vec<Trait>,
    pub heritabilities: Vec<f32>,
    pub models: Vec<GenoPhenoNetwork>,
}

impl fmt::Display for GeneticModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "------------------------")?;
        writeln!(f, "GeneticModel struct")?;
        writeln!(f, "------------------------")?;
        writeln!(f, "\t- Chromosomes: {}", self.genome.len())?;
        writeln!(f, "\t- Loci: {}", self.loci.len())?;
        writeln!(
            f,
            "\t- Locus Alleles: {}",
            self.loci.iter().map(|l| l.col_idx.len()).sum::<usize>()
        )?;
        writeln!(f, "\t- Traits: {}", self.traits.len())?;
        writeln!(f, "\t- Heritabilities: {}", self.heritabilities.len())?;
        writeln!(f, "\t- Genotype-to-phenotype models:")?;
        for (m, t) in self.models.iter().zip(self.traits.iter()) {
            for ((w, b), a) in m.weights.iter().zip(m.biases.iter()).zip(m.activations.iter()) {
                writeln!(f, "\t  ---------------------------------")?;
                writeln!(f, "\t  Trait: {} | Weights: {}", t.name, w)?;
                writeln!(f, "\t  Trait: {} | Biases: {}", t.name, b)?;
                writeln!(f, "\t  Trait: {} | Activations: {}", t.name, a)?;
                writeln!(f, "\t  ---------------------------------")?;
            }
        }
        Ok(())
    }
}

pub fn sim_traits(n_traits: usize) -> Result<Vec<Trait>> {
    ensure!(
        n_traits > 0,
        "The number of traits need to be greater than zero!"
    );
    let mut traits: Vec<Trait> = Vec::with_capacity(n_traits);
    let n_digits: usize = format!("{}", n_traits).len();
    for i in 0..n_traits {
        traits.push(Trait {
            name: format!("trait_{:0>n_digits$}", i),
            description: "".to_owned(),
        });
    }
    Ok(traits)
}

pub fn sim_genotype_to_phenotype_map(
    ctx: &GpuContext,
    genome: &[Chromosome],
    loci: &[Locus],
    traits: &[Trait],
    seed: u64,
) -> Result<GeneticModel> {
    todo!()
}

pub fn calc_phenotypes(
    ctx: &GpuContext,
    genotype_data: &GpuTensor,
    genetic_arch: &GeneticModel,
    ploidy: usize,
    seed: u64,
) -> Result<GpuTensor> {
    todo!()
}

pub fn sim_phenotype_data(
    ctx: &GpuContext,
    genome: &[Chromosome],
    loci: &[Locus],
    genotype_data: &GpuTensor,
    traits: &[Trait],
    ploidy: usize,
    seed: u64,
) -> Result<(GpuTensor, GeneticModel)> {
    let genetic_arch: GeneticModel =
        sim_genotype_to_phenotype_map(ctx, genome, loci, traits, seed)?;
    let phenotype_data = calc_phenotypes(ctx, genotype_data, &genetic_arch, ploidy, seed)?;
    Ok((phenotype_data, genetic_arch))
}

// TODO: implement better genotype-to-phenotype genetic_arch, i.e. allowing for genetic architecture paramter inputs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::*;
    use crate::linalg::context::GpuContext;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    fn mean(x: &[f32]) -> f32 {
        x.iter().sum::<f32>() / (x.len() as f32)
    }

    fn variance(x: &[f32]) -> f32 {
        let mu = mean(x);

        x.iter().map(|v| (*v - mu).powi(2)).sum::<f32>() / (x.len() as f32)
    }

    fn skewness(x: &[f32]) -> f32 {
        let mu = mean(x);
        let sd = variance(x).sqrt();

        x.iter().map(|v| (*v - mu).powi(3)).sum::<f32>() / ((x.len() as f32) * sd.powi(3))
    }

    #[test]
    fn sim_traits_rejects_zero() {
        assert!(sim_traits(0).is_err());
    }

    #[test]
    fn sim_traits_returns_requested_number() {
        let traits = sim_traits(17).unwrap();

        assert_eq!(traits.len(), 17);
    }

    #[test]
    fn sim_traits_initialises_empty_descriptions() {
        let traits = sim_traits(25).unwrap();

        assert!(traits.iter().all(|t| t.description.is_empty()));
    }

    #[test]
    fn sim_traits_names_are_unique() {
        let traits = sim_traits(100).unwrap();

        let mut names: Vec<String> = traits.iter().map(|t| t.name.clone()).collect();

        names.sort();
        names.dedup();

        assert_eq!(names.len(), traits.len());
    }

    #[test]
    fn sim_traits_is_deterministic() {
        let a = sim_traits(100).unwrap();
        let b = sim_traits(100).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn mapper_is_deterministic() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(10).unwrap();

        let a = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        let b = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        assert_eq!(a.genome, b.genome);
        assert_eq!(a.loci, b.loci);
        assert_eq!(a.traits, b.traits);
        assert_eq!(a.heritabilities, b.heritabilities);

        assert_eq!(
            a.allele_effects.to_vec_f32(&ctx).unwrap(),
            b.allele_effects.to_vec_f32(&ctx).unwrap()
        );
    }

    #[test]
    fn mapper_heritabilities_match_trait_count() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(13).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        assert_eq!(genetic_arch.heritabilities.len(), traits.len());
    }

    #[test]
    fn mapper_heritabilities_are_bounded() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(20).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        assert!(
            genetic_arch
                .heritabilities
                .iter()
                .all(|h| (0.0..=1.0).contains(h))
        );
    }

    #[test]
    fn mapper_allele_effect_shape_matches_dimensions() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(7).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        let n_loci_alleles: usize = loci.iter().map(|l| l.col_idx.len()).sum();

        assert_eq!(
            genetic_arch.allele_effects.shape,
            vec![n_loci_alleles as u32, traits.len() as u32]
        );
    }

    #[test]
    fn mapper_contains_non_zero_effects() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(20).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        let effects = genetic_arch.allele_effects.to_vec_f32(&ctx).unwrap();

        assert!(effects.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn calc_phenotypes_returns_expected_shape() {
        let ctx = context();

        let entries = sim_entries(200).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let sexes = sim_sexes(200, false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let traits = sim_traits(8).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 456).unwrap();

        let phenotype_data = calc_phenotypes(&ctx, &genotype_data, &genetic_arch, 2, 789).unwrap();

        assert_eq!(phenotype_data.shape, vec![200, 8]);
    }

    #[test]
    fn calc_phenotypes_is_deterministic() {
        let ctx = context();

        let entries = sim_entries(100).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let sexes = sim_sexes(100, false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let traits = sim_traits(5).unwrap();

        let genetic_arch =
            sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 456).unwrap();

        let a = calc_phenotypes(&ctx, &genotype_data, &genetic_arch, 2, 999).unwrap();

        let b = calc_phenotypes(&ctx, &genotype_data, &genetic_arch, 2, 999).unwrap();

        assert_eq!(a.to_vec_f32(&ctx).unwrap(), b.to_vec_f32(&ctx).unwrap());
    }

    #[test]
    fn sim_phenotype_data_returns_expected_shapes() {
        let ctx = context();

        let entries = sim_entries(250).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let sexes = sim_sexes(250, false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let traits = sim_traits(11).unwrap();

        let (phenotype_data, genetic_arch) =
            sim_phenotype_data(&ctx, &genome, &loci, &genotype_data, &traits, 2, 999).unwrap();

        assert_eq!(phenotype_data.shape, vec![250, 11]);

        assert_eq!(genetic_arch.heritabilities.len(), 11);
    }

    #[test]
    fn simulated_phenotypes_are_approximately_normal() {
        let ctx = context();

        let entries = sim_entries(1_000).unwrap();

        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let sexes = sim_sexes(entries.len(), false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let traits = sim_traits(1).unwrap();

        let (phenotype_data, _) =
            sim_phenotype_data(&ctx, &genome, &loci, &genotype_data, &traits, 2, 999).unwrap();

        let y = phenotype_data.to_vec_f32(&ctx).unwrap();

        let mu = mean(&y);
        let sd = variance(&y).sqrt();

        let within_1 = y.iter().filter(|&&v| (v - mu).abs() <= sd).count() as f32 / y.len() as f32;

        let within_2 =
            y.iter().filter(|&&v| (v - mu).abs() <= 2.0 * sd).count() as f32 / y.len() as f32;

        let within_3 =
            y.iter().filter(|&&v| (v - mu).abs() <= 3.0 * sd).count() as f32 / y.len() as f32;

        assert!((within_1 - 0.6827).abs() < 0.05);
        assert!((within_2 - 0.9545).abs() < 0.05);
        assert!((within_3 - 0.9973).abs() < 0.02);

        assert!(skewness(&y).abs() < 0.2);
    }

    #[test]
    fn simulated_phenotypes_have_non_zero_variance() {
        let ctx = context();

        let entries = sim_entries(500).unwrap();

        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let sexes = sim_sexes(entries.len(), false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let traits = sim_traits(1).unwrap();

        let (phenotype_data, _) =
            sim_phenotype_data(&ctx, &genome, &loci, &genotype_data, &traits, 2, 999).unwrap();

        let y = phenotype_data.to_vec_f32(&ctx).unwrap();

        assert!(variance(&y) > 0.0);
    }
}
