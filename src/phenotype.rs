use crate::genotype::*;
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

#[derive(Debug)]
pub struct MapperG2P {
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub traits: Vec<Trait>,
    pub heritabilities: Vec<f32>,
    pub allele_effects: GpuTensor,
}

impl fmt::Display for MapperG2P {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "------------------------")?;
        writeln!(f, "MapperG2P struct")?;
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
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Allele effects: {}", self.allele_effects)?;
        writeln!(f, "\t  ---------------------------------")?;
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
) -> Result<MapperG2P> {
    let n_loci: usize = loci.len();
    let n_loci_alleles: usize = loci.iter().map(|l| l.col_idx.len()).sum();
    let n_traits: usize = traits.len();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let unif_p: Uniform<usize> = Uniform::new(1, n_loci)
        .expect("Error initialising uniform distribution for number of loci with effects!");
    let unif_eff_mu: Uniform<f32> = Uniform::new(-1.0, 1.0).expect("Error initialising uniform distribution for the mean of normally distributed allele effects!");
    let unif_eff_sd: Uniform<f32> = Uniform::new(1.0, 5.0).expect("Error initialising uniform distribution for the standard deviation of normally distributed allele effects!");
    let unif_h2: Uniform<f32> = Uniform::new(f32::EPSILON, 1.0)
        .expect("Error initialising uniform distribution for the trait hertabilities!");

    let mut heritabilities: Vec<f32> = Vec::with_capacity(n_traits);
    let mut allele_effects_tmp: Vec<f32> = vec![0.0; n_loci_alleles * n_traits];
    for j in 0..n_traits {
        let n_loci_with_effects = unif_p.sample(&mut rng);
        let mean = unif_eff_mu.sample(&mut rng);
        let sd = unif_eff_sd.sample(&mut rng);
        heritabilities.push(unif_h2.sample(&mut rng));
        let normal = Normal::new(mean, sd).expect("Error initialising a normal distribution!");
        let idx_loci_with_effects: Vec<usize> =
            index::sample(&mut rng, n_loci, n_loci_with_effects).into_vec();
        for idx_locus in idx_loci_with_effects {
            for &idx_locus_allele in &loci[idx_locus].col_idx {
                let idx = (idx_locus_allele * n_traits) + j;
                allele_effects_tmp[idx] = normal.sample(&mut rng);
            }
        }
    }
    let allele_effects: GpuTensor = GpuTensor::from_vec_f32(
        ctx,
        &allele_effects_tmp,
        &[n_loci_alleles as u32, n_traits as u32],
        None,
        None,
    )?;
    Ok(MapperG2P {
        genome: genome.to_owned(),
        loci: loci.to_owned(),
        traits: traits.to_owned(),
        heritabilities,
        allele_effects,
    })
}

pub fn calc_phenotypes(
    ctx: &GpuContext,
    genotype_data: &GpuTensor,
    mapper: &MapperG2P,
    ploidy: usize,
    seed: u64,
) -> Result<GpuTensor> {
    let n_entries: usize = genotype_data.shape[0] as usize;
    let n_loci_alleles: usize = genotype_data.shape[1] as usize;
    let n_traits: usize = mapper.allele_effects.shape[1] as usize;
    ensure!(
        n_loci_alleles == mapper.loci.iter().map(|x| x.col_idx.len()).sum::<usize>(),
        "The number of loci-alleles do not match between genotype_data and mapper loci!"
    );
    ensure!(
        n_loci_alleles == mapper.allele_effects.shape[0] as usize,
        "The number of loci-alleles do not match between genotype_data and mapper allele effects!"
    );
    ensure!(
        mapper.heritabilities.len() == n_traits,
        "The number of heritabilities does not match the number of traits!"
    );
    for (i, h2) in mapper.heritabilities.iter().enumerate() {
        ensure!(
            (0.0..=1.0).contains(h2),
            "The heritability of trait {} which should be range from 0.0 to 1.0!",
            i
        );
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let kernel = GpuKernel::new(ctx);
    let y0 = kernel.contract(
        &genotype_data.slice_view(&[(0, n_entries), (0, n_loci_alleles), (0, 1)])?,
        &mapper.allele_effects,
    )?;
    let y1 = kernel.contract(
        &genotype_data.slice_view(&[(0, n_entries), (0, n_loci_alleles), (1, 2)])?,
        &mapper.allele_effects,
    )?;
    let y: GpuTensor = kernel.div_scalar(&kernel.add(&y0, &y1)?, ploidy as f32)?;
    // Simulate error effects dependin on trait heritabilities
    let mut errors_tmp: Vec<f32> = vec![0.0; n_entries * n_traits];
    for (j, &h2) in mapper.heritabilities.iter().enumerate() {
        let y_j = y
            .slice_view(&[(0, n_entries), (j, j + 1)])?
            .to_vec_f32(ctx)?;
        let n = y_j.len() as f32;
        let u = y_j.iter().sum::<f32>() / n;
        let var_y_j = y_j.iter().map(|&x| (u - x).powi(2)).sum::<f32>() / n;
        let sd_err = (var_y_j * ((1.00 / h2) - 1.00)).sqrt() + f32::EPSILON;
        let normal: Normal<f32> =
            Normal::new(0.0, sd_err).expect("Error initialising a standard normal distribution!");
        for i in 0..n_entries {
            let idx = (i * n_traits) + j;
            errors_tmp[idx] = normal.sample(&mut rng);
        }
    }
    let errors: GpuTensor = GpuTensor::from_vec_f32(
        ctx,
        &errors_tmp,
        &[n_entries as u32, n_traits as u32],
        None,
        None,
    )?;
    // Add in the standard normal errors
    kernel.add(&y, &errors)
}

pub fn sim_phenotype_data(
    ctx: &GpuContext,
    genome: &[Chromosome],
    loci: &[Locus],
    genotype_data: &GpuTensor,
    traits: &[Trait],
    ploidy: usize,
    seed: u64,
) -> Result<(GpuTensor, MapperG2P)> {
    let mapper: MapperG2P = sim_genotype_to_phenotype_map(ctx, genome, loci, traits, seed)?;
    let phenotype_data = calc_phenotypes(ctx, genotype_data, &mapper, ploidy, seed)?;
    Ok((phenotype_data, mapper))
}

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

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        assert_eq!(mapper.heritabilities.len(), traits.len());
    }

    #[test]
    fn mapper_heritabilities_are_bounded() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(20).unwrap();

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        assert!(
            mapper
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

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        let n_loci_alleles: usize = loci.iter().map(|l| l.col_idx.len()).sum();

        assert_eq!(
            mapper.allele_effects.shape,
            vec![n_loci_alleles as u32, traits.len() as u32]
        );
    }

    #[test]
    fn mapper_contains_non_zero_effects() {
        let ctx = context();

        let genome = sim_genome(5, false).unwrap();
        let loci = sim_loci(&genome, 100, 42).unwrap();
        let traits = sim_traits(20).unwrap();

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 123).unwrap();

        let effects = mapper.allele_effects.to_vec_f32(&ctx).unwrap();

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

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 456).unwrap();

        let phenotype_data = calc_phenotypes(&ctx, &genotype_data, &mapper, 2, 789).unwrap();

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

        let mapper = sim_genotype_to_phenotype_map(&ctx, &genome, &loci, &traits, 456).unwrap();

        let a = calc_phenotypes(&ctx, &genotype_data, &mapper, 2, 999).unwrap();

        let b = calc_phenotypes(&ctx, &genotype_data, &mapper, 2, 999).unwrap();

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

        let (phenotype_data, mapper) =
            sim_phenotype_data(&ctx, &genome, &loci, &genotype_data, &traits, 2, 999).unwrap();

        assert_eq!(phenotype_data.shape, vec![250, 11]);

        assert_eq!(mapper.heritabilities.len(), 11);
    }

    #[test]
    fn simulated_phenotypes_are_approximately_normal() {
        let ctx = context();

        let entries = sim_entries(10_000).unwrap();

        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

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

        let entries = sim_entries(1000).unwrap();

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
