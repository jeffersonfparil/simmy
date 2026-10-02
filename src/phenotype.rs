use crate::genotype::{Chromosome, Locus};
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
pub struct GeneticModel {
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub traits: Vec<Trait>,
    pub repeatabilities: Vec<f32>,
    // We've opt to generalise the genetic model as a multilayer perceptron or artificial neural network with 0 or more hidden layers translating to a vector of 1 or more GpuTensors of weights and biases.
    // Encapsulates all traits, i.e. the output nodes is of length traits.len() allowing for hidden correlations between traits
    pub weights: Vec<GpuTensor>,
    pub biases: Vec<GpuTensor>,
    pub activations: Vec<Activation>,
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
        writeln!(f, "\t- Repeatabilities: {}", self.repeatabilities.len())?;
        writeln!(f, "\t- Genotype-to-phenotype models:")?;
        for ((w, b), a) in self
            .weights
            .iter()
            .zip(self.biases.iter())
            .zip(self.activations.iter())
        {
            writeln!(f, "\t  ---------------------------------")?;
            writeln!(f, "\t  Weights: {}", w)?;
            writeln!(f, "\t  Biases: {}", b)?;
            writeln!(f, "\t  Activations: {:?}", a)?;
            writeln!(f, "\t  ---------------------------------")?;
        }
        Ok(())
    }
}

impl GpuTensor {
    pub fn activate(&mut self, ctx: &GpuContext, activation: &Activation) -> Result<()> {
        match activation {
            Activation::Linear => (),
            Activation::Sigmoid => {
                let kernel = GpuKernel::new(ctx);
                let denominator = kernel.add_scalar(&kernel.exp(&kernel.neg(self)?)?, 1.0)?;
                self.buffer = kernel.pow_scalar(&denominator, -1.0)?.buffer.clone();
            }
            Activation::ReLU => {
                let kernel = GpuKernel::new(ctx);
                let greater_than_zero = kernel.gt_scalar(self, 0.0)?;
                self.buffer = kernel.mul(self, &greater_than_zero)?.buffer.clone();
            }
        }
        Ok(())
    }
}

impl GeneticModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &GpuContext,
        genome: &[Chromosome],
        loci: &[Locus],
        traits: &[Trait], // Each trait's base phenotype will be automatically correlated (cannot be controlled at the moment==>TODO!) and hence these traits will be sharing a common network properties including sparsity (controlled below by the number of loci with effects)
        repeatabilities: &[f32], // Each trait will have it's own repeatability hence specific error effects distribution
        n_loci_alleles_with_effects: Option<usize>, // number of loci with effects which gets translated to sparsity level, i.e. n_loci_alleles_with_effects / n_loci fo use in simulating non-zero weights
        complexity_level: Option<usize>, // number of hidden layers with zero meaning classical linear model
        n_nodes: Option<Vec<usize>>, // number of nodes per hidden layer + output layer, i.e. last value refer to the number of nodes in the final layer ==> the number of traits
        activations: Option<Vec<Activation>>, // activation functions for each layer, i.e. including the final or output layer (e.g. Activation::Linear for the classical linear model with no hidden layers)
        seed: u64,
    ) -> Result<Self> {
        let n_chromosomes: usize = genome.len();
        let n_loci: usize = loci.len();
        let mut n_loci_alleles: usize = 0;
        for locus in loci {
            n_loci_alleles += locus.col_idx.len();
            ensure!(
                locus.chromosome_id < n_chromosomes,
                "The chromosome id ({}) in loci is out-of-bounds (n_chromosomes={})!",
                locus.chromosome_id,
                n_chromosomes
            );
        }
        let n_traits: usize = traits.len();
        let n_loci_alleles_with_effects: usize = match n_loci_alleles_with_effects {
            Some(x) => x,
            None => 10.min(n_loci_alleles),
        };
        let all_none: bool =
            complexity_level.is_none() && n_nodes.is_none() && activations.is_none();
        let all_some: bool =
            complexity_level.is_some() && n_nodes.is_some() && activations.is_some();
        ensure!(
            all_none || all_some,
            "If complexity_level, n_nodes or activations is none then all of them should be None!"
        );
        let complexity_level: usize = complexity_level.unwrap_or(0);
        let n_nodes: Vec<usize> = match n_nodes {
            Some(x) => x,
            None => vec![n_traits],
        };
        let activations: Vec<Activation> = match activations {
            Some(x) => x,
            None => vec![Activation::Linear],
        };
        ensure!(
            n_chromosomes > 0,
            "The number of chromosomes need to be greater than zero!"
        );
        ensure!(
            n_loci > 0,
            "The number of loci need to be greater than zero!"
        );
        ensure!(
            n_loci >= n_chromosomes,
            "The number of loci need to be at least as many as the number of chromosomes!"
        );
        ensure!(n_traits > 0, "No traits defined!");
        ensure!(
            n_traits == repeatabilities.len(),
            "The number of traits and repeatabilities do not match!"
        );
        ensure!(
            repeatabilities.iter().all(|r| (0.0..=1.0).contains(r)),
            "Repeatabilities should range from 0 to 1!"
        );
        ensure!(
            n_loci_alleles_with_effects <= n_loci_alleles,
            "The number of loci-alleles with effects is more than the number of loci-alleles!"
        );
        ensure!(
            (complexity_level + 1) == n_nodes.len(),
            "The complexity (number of hidden layers) and number nodes per hidden layer + output layer do not match ((complexity_level + 1) != n_nodes.len())!"
        );
        ensure!(
            n_traits == n_nodes[n_nodes.len() - 1],
            "The number of nodes in the output layer does not match the number of traits!"
        );
        ensure!(
            (complexity_level + 1) == activations.len(),
            "The complexity (number of hidden layers) and number activation functions do not match (complexity_level + 1 != activations.len())!"
        );

        let f_effects: f32 = (n_loci_alleles_with_effects as f32) / (n_loci_alleles as f32);

        let mut weights: Vec<GpuTensor> = Vec::with_capacity(complexity_level + 1);
        let mut biases: Vec<GpuTensor> = Vec::with_capacity(complexity_level + 1);

        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(seed);
        let uniform: Uniform<f32> =
            Uniform::new(-1.0, 1.0).expect("Failed to initialise Uniform(-1,2) distribution!");

        // Input layer (which may be the only layer, hence n_nodes=vec![n_loci_alleles], i.e. simple linear model)
        let mut n_rows: usize = n_loci_alleles;
        let mut n_cols: usize = n_nodes[0];
        let n_effects: usize = ((n_rows * n_cols) as f32 * f_effects).round().max(1.0) as usize;
        let idx_effects: Vec<usize> =
            index::sample(&mut rng, n_rows * n_cols, n_effects).into_vec();
        let mut w_vec: Vec<f32> = vec![0.0; n_rows * n_cols];
        for i in idx_effects {
            w_vec[i] = uniform.sample(&mut rng);
        }
        let mut b_vec: Vec<f32> = vec![0.0; n_cols];
        for b_i in b_vec.iter_mut() {
            *b_i = uniform.sample(&mut rng);
        }
        weights.push(GpuTensor::from_vec_f32(
            ctx,
            &w_vec,
            &[n_rows as u32, n_cols as u32],
            None,
            None,
        )?);
        biases.push(GpuTensor::from_vec_f32(
            ctx,
            &b_vec,
            &[1u32, n_cols as u32],
            None,
            None,
        )?);
        for i in 0..complexity_level {
            n_rows = weights[i].shape[1] as usize;
            n_cols = n_nodes[i + 1];
            // Dense hidden layers for simplicity
            w_vec = vec![0.0; n_rows * n_cols];
            for w_i in w_vec.iter_mut() {
                *w_i = uniform.sample(&mut rng);
            }
            b_vec = vec![0.0; n_cols];
            for b_i in b_vec.iter_mut() {
                *b_i = uniform.sample(&mut rng);
            }
            weights.push(GpuTensor::from_vec_f32(
                ctx,
                &w_vec,
                &[n_rows as u32, n_cols as u32],
                None,
                None,
            )?);
            biases.push(GpuTensor::from_vec_f32(
                ctx,
                &b_vec,
                &[1u32, n_cols as u32],
                None,
                None,
            )?);
        }
        Ok(Self {
            genome: genome.to_owned(),
            loci: loci.to_owned(),
            traits: traits.to_owned(),
            repeatabilities: repeatabilities.to_owned(),
            weights,
            biases,
            activations: activations.clone(),
        })
    }
    pub fn check(&self) -> Result<()> {
        let n_chromosomes: usize = self.genome.len();
        let n_loci: usize = self.loci.len();
        let mut n_loci_alleles: usize = 0;
        for locus in &self.loci {
            n_loci_alleles += locus.col_idx.len();
            ensure!(
                locus.chromosome_id < n_chromosomes,
                "The chromosome id ({}) in loci is out-of-bounds (n_chromosomes={})!",
                locus.chromosome_id,
                n_chromosomes
            );
        }
        let n_traits: usize = self.traits.len();
        ensure!(
            n_chromosomes > 0,
            "The number of chromosomes need to be greater than zero!"
        );
        ensure!(
            n_loci > 0,
            "The number of loci need to be greater than zero!"
        );
        ensure!(
            n_loci >= n_chromosomes,
            "The number of loci need to be at least as many as the number of chromosomes!"
        );
        ensure!(n_traits > 0, "No traits defined!");
        ensure!(
            n_traits == self.repeatabilities.len(),
            "The number of traits and repeatabilities do not match!"
        );
        ensure!(
            self.repeatabilities.iter().all(|r| (0.0..=1.0).contains(r)),
            "Repeatabilities should range from 0 to 1!"
        );
        ensure!(
            !self.weights.is_empty(),
            "Model is undefined! No weights found!"
        );
        ensure!(
            n_loci_alleles == self.weights[0].shape[0] as usize,
            "The number of loci-alleles does not match the first weights tensor!"
        );
        ensure!(
            self.weights.len() == self.biases.len(),
            "The weights ({}) and biases ({}) are incompatible!",
            self.weights.len(),
            self.biases.len()
        );
        ensure!(
            self.weights.len() == self.activations.len(),
            "The weights ({}) and activations ({}) are incompatible!",
            self.weights.len(),
            self.activations.len()
        );
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            ensure!(
                w.shape.len() == 2,
                "We expect weights to be 2D tensors! The {}th weight has shape: {:?}!",
                i,
                w.shape
            );
            ensure!(
                b.shape.len() == 2,
                "We expect biases to be 2D tensors! The {}th bias has shape: {:?}!",
                i,
                b.shape
            );
            ensure!(
                b.shape[0] == 1,
                "The first dimension of the {}ith biases should be of length 1!",
                i
            );
            ensure!(
                w.shape[1] == b.shape[1],
                "The {}ith weights and biases are incompatible!",
                i
            );
            if i > 0 {
                ensure!(
                    self.weights[i - 1].shape[1] == w.shape[0],
                    "The {}ith and {}ith weights are incompatible!",
                    i - 1,
                    i
                );
            }
        }
        ensure!(
            self.weights[self.weights.len() - 1].shape[1] as usize == self.traits.len(),
            "The number output nodes does not match the number of traits!"
        );
        Ok(())
    }
    pub fn calc_phenotypes(
        &self,
        ctx: &GpuContext,
        genotype_data: &GpuTensor,
        seed: u64,
    ) -> Result<GpuTensor> {
        self.check()?;
        ensure!(
            genotype_data.shape.len() == 3,
            "We expect the genotype_data to be a 3D tensor (entries x loci-alleles x homologous chromosomes)!"
        );
        ensure!(
            self.weights[0].shape[0] == genotype_data.shape[1],
            "The model (1st weight shape: {:?}) and genotype_data (shape: {:?}) are incompatible!",
            self.weights[0].shape,
            genotype_data.shape
        );
        let n_entries: usize = genotype_data.shape[0] as usize;
        let n_loci_alleles: usize = genotype_data.shape[1] as usize;
        let n_traits: usize = self.traits.len();
        ensure!(
            genotype_data.shape[2] == 2,
            "We expect the 3rd dimension to have 2 levels, one for each homologous chromosome or one from each parent!"
        );
        let kernel = GpuKernel::new(ctx);
        let mut output: GpuTensor = {
            // Note that we are collapsing the allele dosages of the 2 homologous chromosomes and we are not accounting for haplotype effects/phase information for simplicity!
            let x: GpuTensor = kernel.add(
                &genotype_data.slice_view(&[(0, n_entries), (0, n_loci_alleles), (0, 1)])?,
                &genotype_data.slice_view(&[(0, n_entries), (0, n_loci_alleles), (1, 2)])?,
            )?;
            let xw: GpuTensor = kernel.contract(&x, &self.weights[0])?;
            // To allow broadcasting in add
            let n_tmp: usize = xw.shape[0] as usize;
            let p_tmp: usize = xw.shape[1] as usize;
            let mut b_new: GpuTensor = self.biases[0].slice_view(&[(0, 1), (0, p_tmp)])?;
            b_new.shape = vec![n_tmp as u32, p_tmp as u32];
            b_new.strides = vec![0u32, 1u32];
            let mut y = kernel.add(&xw, &b_new)?;
            y.activate(ctx, &self.activations[0])?;
            y
        };
        for ((w, b), a) in self
            .weights
            .iter()
            .zip(self.biases.iter())
            .zip(self.activations.iter())
            .skip(1)
        {
            output = kernel.contract(&output, w)?;
            // To allow broadcasting in add
            let n_tmp: usize = output.shape[0] as usize;
            let p_tmp: usize = output.shape[1] as usize;
            let mut b_new: GpuTensor = b.slice_view(&[(0, 1), (0, p_tmp)])?;
            b_new.shape = vec![n_tmp as u32, p_tmp as u32];
            b_new.strides = vec![0u32, 1u32];
            output = kernel.add(&output, &b_new)?;
            output.activate(ctx, a)?;
        }
        // Add error effects
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut errors_tmp: Vec<f32> = vec![0.0; n_entries * n_traits];
        for j in 0..n_traits {
            let r: f32 = self.repeatabilities[j];
            let y: Vec<f32> = output
                .slice_view(&[(0, n_entries), (j, j + 1)])?
                .to_vec_f32(ctx)?;
            let y_mu: f32 = y.iter().sum::<f32>() / (y.len() as f32);
            let v_y: f32 = y.iter().map(|x| (x - y_mu).powi(2)).sum::<f32>() / (y.len() as f32);
            let s_e: f32 = (v_y * ((1.00 / r) - 1.00)).sqrt().max(f32::EPSILON);
            let d: Normal<f32> = Normal::new(0.0, s_e)
                .unwrap_or_else(|_| panic!("Failed to initialise Normal({}, {})!", 0.0, s_e));
            for i in 0..n_entries {
                errors_tmp[(i * n_traits) + j] = d.sample(&mut rng);
            }
        }
        let errors: GpuTensor = GpuTensor::from_vec_f32(
            ctx,
            &errors_tmp,
            &[n_entries as u32, n_traits as u32],
            None,
            None,
        )?;
        output = kernel.add(&output, &errors)?;
        Ok(output)
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

#[allow(clippy::too_many_arguments)]
pub fn sim_phenotype_data(
    ctx: &GpuContext,
    genotype_data: &GpuTensor,
    genome: &[Chromosome],
    loci: &[Locus],
    traits: &[Trait], // Each trait's base phenotype will be automatically correlated (cannot be controlled at the moment==>TODO!) and hence these traits will be sharing a common network properties including sparsity (controlled below by the number of loci with effects)
    repeatabilities: &[f32], // Each trait will have it's own repeatability hence specific error effects distribution
    n_loci_alleles_with_effects: Option<usize>, // number of loci with effects which gets translated to sparsity level, i.e. n_loci_alleles_with_effects / n_loci fo use in simulating non-zero weights
    complexity_level: Option<usize>, // number of hidden layers with zero meaning classical linear model
    n_nodes: Option<Vec<usize>>, // number of nodes per hidden layer + output layer, i.e. last value refer to the number of nodes in the final layer ==> the number of traits
    activations: Option<Vec<Activation>>, // activation functions for each layer, i.e. including the final or output layer (e.g. Activation::Linear for the classical linear model with no hidden layers)
    seed: u64,
) -> Result<(GpuTensor, GeneticModel)> {
    let genetic_model: GeneticModel = GeneticModel::new(
        ctx,
        genome,
        loci,
        traits,
        repeatabilities,
        n_loci_alleles_with_effects,
        complexity_level,
        n_nodes,
        activations,
        seed,
    )?;
    let phenotype_data: GpuTensor = genetic_model.calc_phenotypes(ctx, genotype_data, seed)?;
    Ok((phenotype_data, genetic_model))
}

// TODO: implement better genotype-to-phenotype genetic_arch, i.e. allowing for genetic architecture paramter inputs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::*;
    use crate::genotype::*;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    fn mean(x: &[f32]) -> f32 {
        x.iter().sum::<f32>() / (x.len() as f32)
    }
    fn variance(x: &[f32]) -> f32 {
        let m = mean(x);
        x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / (x.len() as f32)
    }
    fn skewness(x: &[f32]) -> f32 {
        let m = mean(x);
        let s = variance(x).sqrt();
        x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f32>() / (x.len() as f32)
    }
    #[test]
    fn sim_traits_rejects_zero() {
        assert!(sim_traits(0).is_err());
    }
    #[test]
    fn sim_traits_returns_requested_number() {
        assert_eq!(sim_traits(17).unwrap().len(), 17);
    }
    #[test]
    fn sim_traits_names_are_unique() {
        let traits = sim_traits(100).unwrap();
        let mut set = std::collections::HashSet::new();
        for t in traits.iter() {
            assert!(set.insert(t.name.clone()));
        }
    }
    #[test]
    fn genetic_model_construction_linear() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(3).unwrap();
        let repeatabilities = vec![0.5, 0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        assert_eq!(model.weights.len(), 1);
        assert_eq!(model.biases.len(), 1);
        assert_eq!(model.activations.len(), 1);
        assert!(model.check().is_ok());
    }
    #[test]
    fn genetic_model_construction_hidden_layers() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(2).unwrap();
        let repeatabilities = vec![0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            Some(2),
            Some(vec![16, 8, 2]),
            Some(vec![Activation::ReLU, Activation::ReLU, Activation::Linear]),
            42,
        )
        .unwrap();
        assert_eq!(model.weights.len(), 3);
        assert_eq!(model.biases.len(), 3);
        assert_eq!(model.activations.len(), 3);
        assert!(model.check().is_ok());
    }
    #[test]
    fn repeatabilities_must_be_bounded() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(2).unwrap();
        assert!(
            GeneticModel::new(
                &ctx,
                &genome,
                &loci,
                &traits,
                &[-0.1, 0.5],
                None,
                None,
                None,
                None,
                42
            )
            .is_err()
        );
        assert!(
            GeneticModel::new(
                &ctx,
                &genome,
                &loci,
                &traits,
                &[0.5, 1.5],
                None,
                None,
                None,
                None,
                42
            )
            .is_err()
        );
    }
    #[test]
    fn output_nodes_must_match_traits() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(3).unwrap();
        let repeatabilities = vec![0.5, 0.5, 0.5];
        assert!(
            GeneticModel::new(
                &ctx,
                &genome,
                &loci,
                &traits,
                &repeatabilities,
                None,
                Some(1),
                Some(vec![8, 2]),
                Some(vec![Activation::ReLU, Activation::Linear]),
                42
            )
            .is_err()
        );
    }
    #[test]
    fn genetic_model_is_deterministic_given_seed() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(2).unwrap();
        let repeatabilities = vec![0.5, 0.5];
        let m1 = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let m2 = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        assert_eq!(
            m1.weights[0].to_vec_f32(&ctx).unwrap(),
            m2.weights[0].to_vec_f32(&ctx).unwrap()
        );
    }
    #[test]
    fn first_layer_contains_some_sparsity() {
        let ctx = context();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(2).unwrap();
        let repeatabilities = vec![0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            Some(5),
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let w = model.weights[0].to_vec_f32(&ctx).unwrap();
        assert!(w.contains(&0.0));
    }
    #[test]
    fn phenotype_shape_matches_entries_and_traits() {
        let ctx = context();
        let entries = sim_entries(50).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(4).unwrap();
        let repeatabilities = vec![0.5, 0.5, 0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let phenotype_data = model.calc_phenotypes(&ctx, &genotype_data, 42).unwrap();
        assert_eq!(phenotype_data.shape[0] as usize, entries.len());
        assert_eq!(phenotype_data.shape[1] as usize, traits.len());
    }
    #[test]
    fn phenotype_is_deterministic_given_seed() {
        let ctx = context();
        let entries = sim_entries(50).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(2).unwrap();
        let repeatabilities = vec![0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let p1 = model
            .calc_phenotypes(&ctx, &genotype_data, 999)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        let p2 = model
            .calc_phenotypes(&ctx, &genotype_data, 999)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        assert_eq!(p1, p2);
    }
    #[test]
    fn phenotype_values_are_finite() {
        let ctx = context();
        let entries = sim_entries(500).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 200, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(3).unwrap();
        let repeatabilities = vec![0.5, 0.5, 0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let p = model
            .calc_phenotypes(&ctx, &genotype_data, 1)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        assert!(p.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn phenotype_variance_is_nonzero() {
        let ctx = context();
        let entries = sim_entries(1000).unwrap();
        let genome = sim_genome(10, false).unwrap();
        let loci = sim_loci(&genome, 500, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(1).unwrap();
        let repeatabilities = vec![0.5];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let p = model
            .calc_phenotypes(&ctx, &genotype_data, 42)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        assert!(variance(&p) > 0.0);
    }
    #[test]
    fn lower_repeatability_produces_more_variance() {
        let ctx = context();
        let entries = sim_entries(1000).unwrap();
        let genome = sim_genome(10, false).unwrap();
        let loci = sim_loci(&genome, 500, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(1).unwrap();
        let m1 = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &[0.9],
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let m2 = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &[0.1],
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        let p1 = m1
            .calc_phenotypes(&ctx, &genotype_data, 42)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        let p2 = m2
            .calc_phenotypes(&ctx, &genotype_data, 42)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        assert!(variance(&p2) > variance(&p1));
    }
    #[test]
    fn phenotype_distribution_is_approximately_normal() {
        let ctx = context();
        let entries = sim_entries(500).unwrap();
        let genome = sim_genome(10, false).unwrap();
        let loci = sim_loci(&genome, 1000, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(1).unwrap();
        let repeatabilities = vec![0.1];
        let model = GeneticModel::new(
            &ctx,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            Some(2),
            Some(vec![32, 16, 1]),
            Some(vec![Activation::ReLU, Activation::ReLU, Activation::Linear]),
            42,
        )
        .unwrap();
        let p = model
            .calc_phenotypes(&ctx, &genotype_data, 42)
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();
        let skew = skewness(&p);
        assert!(skew.abs() < 1.0);
    }
    #[test]
    fn sim_phenotype_data_returns_expected_shapes() {
        let ctx = context();
        let entries = sim_entries(25).unwrap();
        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 50, 42).unwrap();
        let sexes = sim_sexes(entries.len(), false, 42).unwrap();
        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 42).unwrap();
        let traits = sim_traits(3).unwrap();
        let repeatabilities = vec![0.5, 0.5, 0.5];
        let (phenotype_data, model) = sim_phenotype_data(
            &ctx,
            &genotype_data,
            &genome,
            &loci,
            &traits,
            &repeatabilities,
            None,
            None,
            None,
            None,
            42,
        )
        .unwrap();
        assert_eq!(phenotype_data.shape[0] as usize, entries.len());
        assert_eq!(phenotype_data.shape[1] as usize, traits.len());
        assert_eq!(model.traits.len(), traits.len());
        assert!(model.check().is_ok());
    }
}
