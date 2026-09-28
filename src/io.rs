use crate::linalg::context::GpuContext;
use crate::linalg::kernel::GpuKernel;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use rand::prelude::IndexedRandom;
use rand::seq::index;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp, Normal, Uniform};
use std::borrow::Cow;
use std::fmt;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, PartialEq)]
pub struct Chromosome {
    pub name: String,
    pub length: usize, // chromosome length (i.e. length of the largest homologous chromosome per pair)
    pub ld_decay_distance: usize, // will be used in mating assuming r(d) = exp(-d/L), where d is the distance between a pair of loci in bases and L is ld_decay_distance.
    pub is_sex_chromosome: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Locus {
    pub chromosome_id: usize, // index of the chromosome containing this locus, which assumes one or more chromosomes are stored in a vector (contiguous/ordered list)
    pub position: usize,      // position in the chromosome
    pub alleles: Vec<String>, // sequence of each allele
    pub length: usize, // maximum size of alleles, i.e. the number of bases of the longest allele
    pub col_idx: Vec<usize>, // The column indices in the main genotype tensor, each referring to an allele
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trait {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub species: String,
    pub group: String,
    pub notes: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sex {
    Hermaphrodite,
    Homogametic,
    Heterogametic,
}

#[derive(Debug)]
pub struct Data {
    pub entries: Vec<Entry>,
    pub ploidy: usize,
    pub sexes: Vec<Sex>,
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub haplotype_persistence_probs: Vec<f32>, // Probability of linkage between adjacent loci computed as: `(-distance / ld_decay_distance).exp().max(0.5)` which ranges from 0.5 (no linkage) to 1.0 (complete linkage). This which starts with 0.5 for the first locus.
    pub traits: Vec<Trait>,
    pub genotype_data: GpuTensor, // 3D tensor with shape: n_entries x n_loci_alleles x maternal+paternal haplotypes
    pub phenotype_data: GpuTensor, // 2D tensor with shape: n_entries x n_traits (additionally monogametic = 0.0 and heterogametic = 1.0)
}

#[derive(Debug)]
pub struct MapperG2P {
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub traits: Vec<Trait>,
    pub heritabilities: Vec<f32>,
    pub allele_effects: GpuTensor,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_loci_alleles: usize = self.loci.iter().map(|l| l.col_idx.len()).sum();
        writeln!(f, "Data")?;
        writeln!(f, "\t- Entries: {}", self.entries.len())?;
        writeln!(f, "\t- Ploidy: {}X (always even ploidy)", self.ploidy)?;
        writeln!(f, "\t- Sex:")?;
        writeln!(
            f,
            "\t\t+ Hermaphrodites: {}",
            self.sexes
                .iter()
                .filter(|&x| x == &Sex::Hermaphrodite)
                .count()
        )?;
        writeln!(
            f,
            "\t\t+ Homogametics: {}",
            self.sexes
                .iter()
                .filter(|&x| x == &Sex::Homogametic)
                .count()
        )?;
        writeln!(
            f,
            "\t\t+ Heterogametics: {}",
            self.sexes
                .iter()
                .filter(|&x| x == &Sex::Heterogametic)
                .count()
        )?;
        writeln!(f, "\t- Chromosomes: {}", self.genome.len())?;
        writeln!(f, "\t- Loci: {}", self.loci.len())?;
        writeln!(f, "\t- Locus Alleles: {}", n_loci_alleles)?;
        writeln!(f, "\t- Haplotype persistence probability")?;
        writeln!(
            f,
            "\t  (probabilities of adjacent loci pairs being inherited together): {}",
            self.haplotype_persistence_probs.len()
        )?;
        writeln!(f, "\t- Traits: {}", self.traits.len())?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Genotype Tensor Shape: {}", self.genotype_data)?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Phenotype Tensor Shape: {}", self.phenotype_data)?;
        writeln!(f, "\t  ---------------------------------")?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WGSLMeiosisParams {
    /// The total number of allele columns across all loci in the genotype tensor.
    n_loci_alleles: u32,
    /// The total number of loci being simulated.
    n_loci: u32,
    /// The base random seed used to initialize the thread-local PCG random number generator.
    seed: u32,
    /// Explicit padding to ensure the struct aligns to a 16-byte boundary,
    /// which is strictly required for WGSL uniform buffers.
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WGSLMatingPairData {
    /// The row index of the first parent in the parental genotype tensor.
    p1_idx: u32,
    /// The row index of the second parent in the parental genotype tensor.
    p2_idx: u32,
    /// The biological sex of the first parent.
    /// Encoded as: 0 = Hermaphrodite, 1 = Homogametic, 2 = Heterogametic.
    sex_p1: u32,
    /// The biological sex of the second parent.
    /// Encoded as: 0 = Hermaphrodite, 1 = Homogametic, 2 = Heterogametic.
    sex_p2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WGSLLocusData {
    /// The inclusive starting column index for this locus in the genotype tensor.
    start_col: u32,
    /// The exclusive ending column index for this locus in the genotype tensor.
    end_col: u32,
    /// The linkage probability relative to the previous locus.
    /// `1.0` implies complete linkage (no crossover), while `0.5` implies
    /// independent assortment (50% chance of crossover).
    r: f32,
    /// Bitwise flags representing locus characteristics.
    /// * Bit 0 (`1`): Indicates if the locus resides on a sex chromosome.
    /// * Bit 1 (`2`): Indicates if the locus is the start of a new chromosome
    ///   (forcing independent assortment).
    metadata: u32,
}

pub fn sim_entries(n_entries: usize) -> Result<Vec<Entry>> {
    ensure!(
        n_entries > 0,
        "The number of entries need to be greater than zero!"
    );
    let mut entries: Vec<Entry> = Vec::with_capacity(n_entries);
    let n_digits: usize = format!("{}", n_entries - 1).len();
    for i in 0..n_entries {
        entries.push(Entry {
            name: format!("entry_{:0>n_digits$}", i),
            species: "".to_owned(),
            group: "".to_owned(),
            notes: "".to_owned(),
        });
    }
    Ok(entries)
}

pub fn sim_genome(n_chromosomes: usize, with_sex: bool) -> Result<Vec<Chromosome>> {
    ensure!(
        n_chromosomes > 0,
        "The number of chromosomes need to be greater than zero!"
    );
    let mut genome: Vec<Chromosome> = Vec::with_capacity(n_chromosomes);
    let n_digits: usize = format!("{}", n_chromosomes - 1).len();
    for i in 0..n_chromosomes {
        genome.push(Chromosome {
            name: format!("chromosome_{:0>n_digits$}", i),
            length: 1_000_000,
            ld_decay_distance: 10_000,
            is_sex_chromosome: with_sex && (i == (n_chromosomes - 1)),
        });
    }
    Ok(genome)
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

pub fn sim_sexes(n_entries: usize, with_sex: bool, seed: u64) -> Result<Vec<Sex>> {
    let mut sexes: Vec<Sex> = Vec::with_capacity(n_entries);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..n_entries {
        let sex = if !with_sex {
            Sex::Hermaphrodite
        } else {
            if rng.random_bool(0.5) {
                Sex::Homogametic
            } else {
                Sex::Heterogametic
            }
        };
        sexes.push(sex);
    }
    Ok(sexes)
}

pub fn sim_loci(genome: &[Chromosome], n_loci: usize, seed: u64) -> Result<Vec<Locus>> {
    ensure!(
        n_loci > 0,
        "The number of loci need to be greater than zero!"
    );
    let n_chromosomes = genome.len();
    ensure!(
        n_loci >= n_chromosomes,
        "The number of loci need to be at least as many as the number of chromosomes!"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Divy up the loci into chromosomes
    let mut positions_per_chromosome: Vec<Vec<usize>> = Vec::with_capacity(n_chromosomes);
    let n_chrom_base = n_loci / n_chromosomes;
    let n_chrom_remainder = n_loci % n_chromosomes;
    for (i, chromosome) in genome.iter().enumerate() {
        let m = if i < n_chrom_remainder {
            n_chrom_base + 1
        } else {
            n_chrom_base
        };
        let n = chromosome.length;
        let mut pos: Vec<usize> = Vec::with_capacity(m);
        for j in 0..m {
            pos.push(j * n / m);
        }
        positions_per_chromosome.push(pos);
    }
    // Simulate loci-alleles
    // TODO: add more alleles but for now we only have SNPs, i.e. A, T, C, G, and D (note that these are still SNPs even though we can have multi-allelic loci because the variations remain single nucleotides)
    let choice_of_alleles = &["A", "T", "C", "G", "D"];
    let mut loci: Vec<Locus> = Vec::with_capacity(n_loci);
    let mut n_loci_alleles: usize = 0;
    let exponential = Exp::new(1.0).expect("Failed to initialise exponential distribution!");
    for (i, positions) in positions_per_chromosome.iter().enumerate() {
        for &pos in positions {
            let n_alleles: usize = if !genome[i].is_sex_chromosome {
                let a: f64 = exponential.sample(&mut rng) + 2.0;
                a.round().min(5.0) as usize
            } else {
                2
            };
            loci.push(Locus {
                chromosome_id: i,
                position: pos,
                alleles: choice_of_alleles[0..n_alleles]
                    .iter()
                    .map(|&x| x.to_owned())
                    .collect::<Vec<String>>(),
                length: 1,
                col_idx: (n_loci_alleles..(n_loci_alleles + n_alleles)).collect::<Vec<usize>>(),
            });
            n_loci_alleles += n_alleles;
        }
    }
    Ok(loci)
}

pub fn haplotype_persistence_probabilities(
    genome: &[Chromosome],
    loci: &[Locus],
) -> Result<Vec<f32>> {
    // Chromosome-specific `ld_decay_distance` is the effective haplotype persistence distance:
    // r(d) = exp(-d / L)
    // where:
    // d = physical distance between adjacent loci
    // L = ld_decay_distance
    //
    // Larger values of L produce longer haplotype blocks and stronger
    // inheritance of neighbouring loci together, while smaller values
    // produce shorter blocks and more frequent recombination events.
    //
    // The resulting linkage probability is bounded to [0.5, 1.0]:
    // 1.0 -> complete linkage (no crossover)
    // 0.5 -> independent assortment
    let n_loci = loci.len();
    let mut haplotype_persistence_probs: Vec<f32> = Vec::with_capacity(n_loci);
    for j in 0..n_loci {
        let idx_locus_1 = if j == 0 { 0 } else { j - 1 };
        let idx_locus_2 = j;
        let r: f32 =
            if (j == 0) || (loci[idx_locus_1].chromosome_id != loci[idx_locus_2].chromosome_id) {
                0.5
            } else {
                let idx_chromosome: usize = loci[idx_locus_1].chromosome_id;
                let ld_decay_distance: f64 = genome[idx_chromosome].ld_decay_distance as f64;
                let distance: f64 = {
                    let position_1: usize = loci[idx_locus_1].position;
                    let position_2: usize = loci[idx_locus_2].position;
                    (position_1 as f64 - position_2 as f64).abs()
                };
                (-distance / ld_decay_distance).exp().max(0.5) as f32 // ranges from 0.5 (no linkage) to 1.0 (complete linkage)
            };
        haplotype_persistence_probs.push(r);
    }
    Ok(haplotype_persistence_probs)
}

pub fn sim_genotype_data(
    ctx: &GpuContext,
    entries: &[Entry],
    genome: &[Chromosome],
    loci: &[Locus],
    sexes: &[Sex],
    ploidy: usize,
    seed: u64,
) -> Result<GpuTensor> {
    let n_entries: usize = entries.len();
    let n_sexes: usize = sexes.len();
    let n_chromosomes: usize = genome.len();
    let n_loci: usize = loci.len();
    let n_loci_alleles: usize = loci.iter().map(|x| x.col_idx.len()).sum();
    ensure!(
        n_entries > 0,
        "The number of entries need to be greater than zero!"
    );
    ensure!(
        n_entries == n_sexes,
        "The number of entries and their sexes do not match!"
    );
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

    ensure!(ploidy > 0, "Ploidy need to be greater than zero!");
    ensure!(
        ploidy.is_multiple_of(2),
        "We do not support odd ploidies at the moment!"
    );
    // Genotype data
    // Note that we are not simulating LD between loci pairs yet, i.e. assuming the resulting alleles are historical effects
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut genotype_data_tmp: Vec<f32> = vec![0.0; n_entries * n_loci_alleles * 2];
    let beta_n: Beta<f32> =
        Beta::new(2.0, 5.0).expect("Failed to initialise a Beta distribution (a=2; b=5)!"); // We will sample allele indices per locus from a skewed distribution, i.e. biased towards the first allele which is more realistic than uniform distribution.
    let beta_u: Beta<f32> =
        Beta::new(0.5, 0.5).expect("Failed to initialise a Beta distribution (a=b=0.5)!"); // We will sample from a realistic U shaped distribution for genomewide loci on a single population.
    for (i, &sex) in sexes.iter().enumerate() {
        let idx_entry = i * 2 * n_loci_alleles;
        for locus in loci {
            if !genome[locus.chromosome_id].is_sex_chromosome {
                // Autosomal loci
                let n_alleles = locus.col_idx.len();
                // Sample alleles and their dosages per homologous chromosome or parent,
                // where the same allele may be sampled and hence fixed on a homologous chromosome and so we add to the initialised 0.0 dosages.
                for j in 0..2 {
                    // Each allele for each ploidy is sampled here
                    for _ in 0..(ploidy / 2) {
                        let idx_allele_1: usize = locus.col_idx
                            [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                        let idx_allele_2: usize = locus.col_idx
                            [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                        let allele_1_dosage_parent_j = beta_u.sample(&mut rng).round();
                        let allele_2_dosage_parent_j = 1.00 - allele_1_dosage_parent_j;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_1) + j] +=
                            allele_1_dosage_parent_j;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_2) + j] +=
                            allele_2_dosage_parent_j;
                    }
                }
            } else {
                // Sex chromosome loci
                // TODO: probably implement a more complex structure where groups of loci are define the sex and not sweeping like this!
                let idx_allele_1: usize = locus.col_idx[0];
                let idx_allele_2: usize = locus.col_idx[1];
                // Sex chromosomes are currently represented by a fixed biallelic locus.
                // Allele 1 corresponds to the shared sex chromosome (e.g. X),
                // while Allele 2 corresponds to the alternate sex chromosome (e.g. Y).
                //
                // The genotype tensor stores two parental haplotypes in its third dimension.
                // For sex chromosomes we initialise deterministic haplotypes rather than
                // sampling alleles, ensuring that homogametic and heterogametic individuals
                // have distinguishable sex-chromosome configurations.
                //
                // This is a simplified representation intended only to model inheritance of
                // sex chromosomes during meiosis and mating. It should not be interpreted as
                // a biologically realistic simulation of full sex-chromosome sequence
                // variation, pseudoautosomal regions, dosage compensation, or copy-number
                // differences.
                match sex {
                    Sex::Homogametic => {
                        // Homogametic individuals (e.g. XX or ZZ):
                        genotype_data_tmp[idx_entry + (2 * idx_allele_1)] += (ploidy / 2) as f32;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_1) + 1] +=
                            (ploidy / 2) as f32;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_2)] += 0.0;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_2) + 1] += 0.0;
                    }
                    Sex::Heterogametic => {
                        // Heterogametic individuals (e.g. XY or ZW).
                        genotype_data_tmp[idx_entry + (2 * idx_allele_1)] += (ploidy / 2) as f32;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_1) + 1] += 0.0;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_2)] += 0.0;
                        genotype_data_tmp[idx_entry + (2 * idx_allele_2) + 1] +=
                            (ploidy / 2) as f32;
                    }
                    Sex::Hermaphrodite => bail!(
                        "Hermaphrodite sexes are not expected because we have sex chromosomes!"
                    ),
                }
            }
        }
    }
    GpuTensor::from_vec_f32(
        ctx,
        &genotype_data_tmp,
        &[n_entries as u32, n_loci_alleles as u32, 2],
        None,
        None,
    )
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
            (0.0..1.0).contains(h2),
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
    let mapper: MapperG2P = sim_genotype_to_phenotype_map(ctx, &genome, &loci, &traits, seed)?;
    let phenotype_data = calc_phenotypes(ctx, &genotype_data, &mapper, ploidy, seed)?;
    Ok((phenotype_data, mapper))
}

impl Data {
    // TODO: make a computation-light init and then a founder population simulator more similar to whar we do with the `new()` method below...
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &GpuContext,
        n_entries: usize,
        n_chromosomes: usize,
        n_loci: usize,
        n_traits: usize,
        ploidy: usize,
        with_sex: bool,
        seed: u64,
    ) -> Result<(Self, MapperG2P)> {
        let entries = sim_entries(n_entries)?;
        let genome = sim_genome(n_chromosomes, with_sex)?;
        let traits = sim_traits(n_traits)?;
        let sexes = sim_sexes(n_entries, with_sex, seed + 1)?;
        let loci = sim_loci(&genome, n_loci, seed + 2)?;
        let haplotype_persistence_probs = haplotype_persistence_probabilities(&genome, &loci)?;
        let genotype_data =
            sim_genotype_data(ctx, &entries, &genome, &loci, &sexes, ploidy, seed + 3)?;
        let (phenotype_data, mapper) = sim_phenotype_data(
            ctx,
            &genome,
            &loci,
            &genotype_data,
            &traits,
            ploidy,
            seed + 4,
        )?;
        // Output
        Ok((
            Self {
                entries,
                ploidy,
                sexes,
                genome,
                loci,
                haplotype_persistence_probs,
                traits,
                genotype_data,
                phenotype_data,
            },
            mapper,
        ))
    }
    pub fn check_dimensions(&self) -> Result<()> {
        let n_entries: usize = self.entries.len();
        let n_loci: usize = self.loci.len();
        let n_haplotype_persistence_probs: usize = self.haplotype_persistence_probs.len();
        let n_loci_alleles: usize = self.loci.iter().map(|l| l.col_idx.len()).sum();
        let n_traits: usize = self.traits.len();
        let n_chromosomes: usize = self.genome.len();
        ensure!(
            n_entries > 0,
            "The number of entries need to be greater than zero!"
        );
        ensure!(
            self.sexes.len() == n_entries,
            "Number of sexes need to match the number of entries!"
        );
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
        ensure!(
            n_loci == n_haplotype_persistence_probs,
            "The number of loci and haplotype_persistence_probabilities do not match!"
        );
        ensure!(
            n_traits > 0,
            "The number of traits need to be greater than zero!"
        );
        ensure!(
            n_entries == self.genotype_data.shape[0] as usize,
            "The number of entries in `entries` and `genotype_data` do not match!"
        );
        ensure!(
            n_entries == self.phenotype_data.shape[0] as usize,
            "The number of entries in `entries` and `phenotype_data` do not match!"
        );
        ensure!(
            n_loci <= self.genotype_data.shape[1] as usize,
            "The number of loci in `loci` must less than or equal to the number of loci-alleles in `genotype_data`!"
        );
        ensure!(
            n_loci_alleles == self.genotype_data.shape[1] as usize,
            "The number of locus alleles in `loci` and `genotype_data` do not match!"
        );
        ensure!(
            n_traits == self.phenotype_data.shape[1] as usize,
            "The number of traits in `traits` and `phenotype_data` do not match!"
        );
        Ok(())
    }
    pub fn sample_mating_pairs(
        &self,
        n_offsprings: usize,
        seed: u64,
    ) -> Result<Vec<(usize, usize)>> {
        self.check_dimensions()?;
        let idx_homogametics_or_hermaphrodites: Vec<usize> = self
            .sexes
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                if (x == Sex::Homogametic) || (x == Sex::Hermaphrodite) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let idx_heterogametics_or_hermaphrodites: Vec<usize> = self
            .sexes
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                if (x == Sex::Heterogametic) || (x == Sex::Hermaphrodite) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        ensure!(
            !idx_homogametics_or_hermaphrodites.is_empty(),
            "There should be hermaphroditic and/or homogametic entries!"
        );
        ensure!(
            !idx_heterogametics_or_hermaphrodites.is_empty(),
            "There should be hermaphroditic and/or heterogametic entries!"
        );
        let mut mating_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_offsprings);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..n_offsprings {
            if let (Some(&x), Some(&y)) = (
                idx_homogametics_or_hermaphrodites.choose(&mut rng),
                idx_heterogametics_or_hermaphrodites.choose(&mut rng),
            ) {
                mating_pairs.push((x, y));
            }
        }
        Ok(mating_pairs)
    }
    pub fn mate(
        &self,
        ctx: &GpuContext,
        mating_pairs: Vec<(usize, usize)>,
        mapper: &MapperG2P,
        seed: u64,
    ) -> Result<Self> {
        self.check_dimensions()?;
        let n_offsprings = mating_pairs.len();
        let n_loci = self.loci.len();
        let n_loci_alleles = self.loci.iter().fold(0, |sum, x| sum + x.col_idx.len());
        ensure!(
            self.genome == mapper.genome,
            "The genome of self and mapper do not match!"
        );
        ensure!(
            self.loci == mapper.loci,
            "The loci of self and mapper do not match!"
        );
        ensure!(
            self.traits == mapper.traits,
            "The traits of self and mapper do not match!"
        );
        // Locus Data
        let mut locus_data_packed = Vec::with_capacity(n_loci);
        for j in 0..n_loci {
            let start_col = *self.loci[j].col_idx.first().unwrap() as u32;
            let end_col = (*self.loci[j].col_idx.last().unwrap() as u32) + 1;
            let mut metadata = 0u32;
            let chr_id = self.loci[j].chromosome_id;
            if self.genome[chr_id].is_sex_chromosome {
                metadata |= 1;
            }
            if j == 0 || chr_id != self.loci[j - 1].chromosome_id {
                metadata |= 2;
            }
            locus_data_packed.push(WGSLLocusData {
                start_col,
                end_col,
                r: self.haplotype_persistence_probs[j],
                metadata,
            });
        }
        // Mating Pair Data
        let mut pairs_data_packed = Vec::with_capacity(n_offsprings);
        for &(p1, p2) in mating_pairs.iter() {
            let sex_p1 = match self.sexes[p1] {
                Sex::Hermaphrodite => 0,
                Sex::Homogametic => 1,
                Sex::Heterogametic => 2,
            };
            let sex_p2 = match self.sexes[p2] {
                Sex::Hermaphrodite => 0,
                Sex::Homogametic => 1,
                Sex::Heterogametic => 2,
            };
            pairs_data_packed.push(WGSLMatingPairData {
                p1_idx: p1 as u32,
                p2_idx: p2 as u32,
                sex_p1,
                sex_p2,
            });
        }
        // GPU Buffers
        let device = &ctx.device;
        let mating_pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mating_pairs"),
            contents: bytemuck::cast_slice(&pairs_data_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let locus_data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("locus_data"),
            contents: bytemuck::cast_slice(&locus_data_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = WGSLMeiosisParams {
            n_loci_alleles: n_loci_alleles as u32,
            n_loci: n_loci as u32,
            seed: seed as u32,
            _padding: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let offspring_size =
            (n_offsprings * n_loci_alleles * 2 * std::mem::size_of::<f32>()) as u64;
        let offspring_genotypes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offspring_genotypes"),
            size: offspring_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Compile and Configure Pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Meiosis Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("meiosis.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Meiosis Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Meiosis Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.genotype_data.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offspring_genotypes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mating_pairs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: locus_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        // GPU Computation
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Meiosis Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Meiosis Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (n_offsprings as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        // Output
        let (mut offsprings, _mapper_drop) = Data::new(
            ctx,
            n_offsprings,
            self.genome.len(),
            n_loci,
            self.traits.len(),
            self.ploidy,
            self.sexes[0] != Sex::Hermaphrodite,
            seed,
        )?;
        offsprings.genome = self.genome.clone();
        offsprings.loci = self.loci.clone();
        offsprings.haplotype_persistence_probs = self.haplotype_persistence_probs.clone();
        offsprings.traits = self.traits.clone();
        for (i, &(p1, p2)) in mating_pairs.iter().enumerate() {
            offsprings.entries[i] = self.entries[p1].clone();
            offsprings.entries[i].name =
                format!("{}--x--{}", self.entries[p1].name, self.entries[p2].name);
        }
        offsprings.genotype_data = GpuTensor::from_buffer(
            std::sync::Arc::new(offspring_genotypes_buf),
            &[n_offsprings as u32, n_loci_alleles as u32, 2],
            None,
            None,
        )?;
        offsprings.phenotype_data =
            calc_phenotypes(ctx, &offsprings.genotype_data, &mapper, self.ploidy, seed)?;
        Ok(offsprings)
    }
    pub fn sim_population() {
        todo!()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::linalg::context::GpuContext;

//     fn context() -> GpuContext {
//         pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
//     }

//     #[test]
//     fn sim_entries_rejects_zero() {
//         assert!(sim_entries(0).is_err());
//     }

//     #[test]
//     fn sim_entries_creates_expected_names() {
//         let entries = sim_entries(10).unwrap();
//         assert_eq!(entries.len(), 10);
//         assert_eq!(entries[0].name, "entry_0");
//         assert_eq!(entries[9].name, "entry_9");
//     }

//     #[test]
//     fn sim_entries_initialises_empty_metadata() {
//         let entries = sim_entries(5).unwrap();
//         for entry in entries {
//             assert!(entry.species.is_empty());
//             assert!(entry.group.is_empty());
//             assert!(entry.notes.is_empty());
//         }
//     }

//     #[test]
//     fn sim_genome_rejects_zero_chromosomes() {
//         assert!(sim_genome(0, false).is_err());
//     }

//     #[test]
//     fn sim_genome_creates_expected_number_of_chromosomes() {
//         let genome = sim_genome(5, false).unwrap();
//         assert_eq!(genome.len(), 5);
//     }

//     #[test]
//     fn sim_genome_assigns_default_properties() {
//         let genome = sim_genome(5, false).unwrap();
//         for chr in genome {
//             assert_eq!(chr.length, 1_000_000);
//             assert_eq!(chr.ld_decay_distance, 10_000);
//         }
//     }

//     #[test]
//     fn sim_genome_marks_last_chromosome_as_sex_chromosome() {
//         let genome = sim_genome(5, true).unwrap();
//         assert_eq!(genome.iter().filter(|c| c.is_sex_chromosome).count(), 1);
//         assert!(genome[4].is_sex_chromosome);
//     }

//     #[test]
//     fn sim_traits_rejects_zero() {
//         assert!(sim_traits(0).is_err());
//     }

//     #[test]
//     fn sim_traits_creates_expected_number() {
//         let traits = sim_traits(7).unwrap();
//         assert_eq!(traits.len(), 7);
//     }

//     #[test]
//     fn sim_traits_initialises_empty_descriptions() {
//         let traits = sim_traits(10).unwrap();
//         assert!(traits.iter().all(|t| t.description.is_empty()));
//     }

//     #[test]
//     fn sim_sexes_is_deterministic() {
//         let a = sim_sexes(100, true, 42).unwrap();
//         let b = sim_sexes(100, true, 42).unwrap();
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn sim_sexes_returns_all_hermaphrodites_when_disabled() {
//         let sexes = sim_sexes(100, false, 42).unwrap();
//         assert!(sexes.iter().all(|s| *s == Sex::Hermaphrodite));
//     }

//     #[test]
//     fn sim_sexes_produces_both_sexes_when_enabled() {
//         let sexes = sim_sexes(1000, true, 42).unwrap();
//         assert!(sexes.contains(&Sex::Homogametic));
//         assert!(sexes.contains(&Sex::Heterogametic));
//     }

//     #[test]
//     fn sim_loci_rejects_zero_loci() {
//         let genome = sim_genome(5, false).unwrap();
//         assert!(sim_loci(&genome, 0, 42).is_err());
//     }

//     #[test]
//     fn sim_loci_rejects_fewer_loci_than_chromosomes() {
//         let genome = sim_genome(10, false).unwrap();
//         assert!(sim_loci(&genome, 5, 42).is_err());
//     }

//     #[test]
//     fn sim_loci_is_deterministic() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci_a = sim_loci(&genome, 100, 42).unwrap();
//         let loci_b = sim_loci(&genome, 100, 42).unwrap();
//         assert_eq!(loci_a.len(), loci_b.len());
//         for (a, b) in loci_a.iter().zip(loci_b.iter()) {
//             assert_eq!(a.chromosome_id, b.chromosome_id);
//             assert_eq!(a.position, b.position);
//             assert_eq!(a.alleles, b.alleles);
//             assert_eq!(a.col_idx, b.col_idx);
//         }
//     }

//     #[test]
//     fn sim_loci_assigns_every_chromosome_at_least_one_locus() {
//         let genome = sim_genome(10, false).unwrap();
//         let loci = sim_loci(&genome, 10, 42).unwrap();
//         let mut counts = [0usize; 10];
//         for locus in loci {
//             counts[locus.chromosome_id] += 1;
//         }
//         assert!(counts.iter().all(|&x| x > 0));
//     }

//     #[test]
//     fn sim_loci_positions_are_sorted_within_chromosomes() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         for chr in 0..genome.len() {
//             let positions: Vec<_> = loci
//                 .iter()
//                 .filter(|x| x.chromosome_id == chr)
//                 .map(|x| x.position)
//                 .collect();
//             assert!(positions.windows(2).all(|w| w[0] <= w[1]));
//         }
//     }

//     #[test]
//     fn sim_loci_column_indices_are_contiguous() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let mut all: Vec<usize> = loci
//             .iter()
//             .flat_map(|x| x.col_idx.iter())
//             .copied()
//             .collect();
//         all.sort_unstable();
//         for (i, idx) in all.iter().enumerate() {
//             assert_eq!(*idx, i);
//         }
//     }

//     #[test]
//     fn sim_loci_allele_counts_match_columns() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         for locus in loci {
//             assert_eq!(locus.alleles.len(), locus.col_idx.len());
//         }
//     }

//     #[test]
//     fn haplotype_persistence_probabilities_match_loci_count() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();
//         assert_eq!(probs.len(), loci.len());
//     }

//     #[test]
//     fn haplotype_persistence_probabilities_are_bounded() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();
//         assert!(probs.iter().all(|&p| (0.5..=1.0).contains(&p)));
//     }

//     #[test]
//     fn haplotype_persistence_first_probability_is_half() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();
//         assert_eq!(probs[0], 0.5);
//     }

//     #[test]
//     fn haplotype_persistence_is_half_at_chromosome_boundaries() {
//         let genome = sim_genome(5, false).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();
//         for j in 1..loci.len() {
//             if loci[j].chromosome_id != loci[j - 1].chromosome_id {
//                 assert_eq!(probs[j], 0.5);
//             }
//         }
//     }

//     #[test]
//     fn haplotype_persistence_increases_monotonically_with_ld_decay_distance() {
//         let genome_small = sim_genome(1, false).unwrap();
//         let mut genome_large = genome_small.clone();
//         genome_large[0].ld_decay_distance = 1_000_000;
//         let loci = sim_loci(&genome_small, 100, 42).unwrap();
//         let p_small = haplotype_persistence_probabilities(&genome_small, &loci).unwrap();
//         let p_large = haplotype_persistence_probabilities(&genome_large, &loci).unwrap();
//         for (large, small) in p_large.iter().zip(p_small.iter()) {
//             assert!(large >= small);
//         }
//     }

//     #[test]
//     fn sim_genotype_data_rejects_zero_entries() {
//         let ctx = context();
//         let genome = sim_genome(1, false).unwrap();
//         let loci = sim_loci(&genome, 1, 42).unwrap();
//         assert!(sim_genotype_data(&ctx, &[], &genome, &loci, &[], 2, 42).is_err());
//     }

//     #[test]
//     fn sim_genotype_data_rejects_zero_ploidy() {
//         let ctx = context();
//         let entries = sim_entries(10).unwrap();
//         let genome = sim_genome(1, false).unwrap();
//         let loci = sim_loci(&genome, 10, 42).unwrap();
//         let sexes = sim_sexes(10, false, 42).unwrap();
//         assert!(sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 0, 42).is_err());
//     }

//     #[test]
//     fn sim_genotype_data_is_deterministic() {
//         let ctx = context();
//         let entries = sim_entries(20).unwrap();
//         let genome = sim_genome(5, true).unwrap();
//         let loci = sim_loci(&genome, 100, 42).unwrap();
//         let sexes = sim_sexes(20, true, 42).unwrap();
//         let g1 = sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();
//         let g2 = sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();
//         assert_eq!(g1.to_vec_f32(&ctx).unwrap(), g2.to_vec_f32(&ctx).unwrap());
//     }

//     #[test]
//     fn sim_phenotype_data_is_deterministic() {
//         let ctx = context();
//         let entries = sim_entries(100).unwrap();
//         let traits = sim_traits(10).unwrap();
//         let p1 = sim_phenotype_data(&ctx, &entries, &traits, 42).unwrap();
//         let p2 = sim_phenotype_data(&ctx, &entries, &traits, 42).unwrap();
//         assert_eq!(p1.to_vec_f32(&ctx).unwrap(), p2.to_vec_f32(&ctx).unwrap());
//     }

//     #[test]
//     fn data_new_rejects_invalid_inputs() {
//         let ctx = context();
//         assert!(Data::new(&ctx, 0, 1, 1, 1, 2, false, 42).is_err());
//         assert!(Data::new(&ctx, 1, 0, 1, 1, 2, false, 42).is_err());
//         assert!(Data::new(&ctx, 1, 1, 0, 1, 2, false, 42).is_err());
//         assert!(Data::new(&ctx, 1, 1, 1, 0, 2, false, 42).is_err());
//         assert!(Data::new(&ctx, 1, 1, 1, 1, 0, false, 42).is_err());
//         assert!(Data::new(&ctx, 1, 1, 1, 1, 3, false, 42).is_err());
//     }

//     #[test]
//     fn check_dimensions_passes() {
//         let ctx = context();
//         let data = Data::new(&ctx, 10, 5, 100, 3, 4, true, 42).unwrap();
//         assert!(data.check_dimensions().is_ok());
//     }

//     #[test]
//     fn check_dimensions_fails_when_haplotype_persistence_probs_are_corrupted() {
//         let ctx = context();
//         let mut data = Data::new(&ctx, 10, 5, 100, 3, 4, true, 42).unwrap();
//         data.haplotype_persistence_probs.pop();
//         assert!(data.check_dimensions().is_err());
//     }

//     #[test]
//     fn sample_mating_pairs_returns_correct_number_and_is_deterministic() {
//         let ctx = context();
//         let data = Data::new(&ctx, 100, 5, 100, 1, 2, true, 42).unwrap();
//         let a = data.sample_mating_pairs(75, 123).unwrap();
//         let b = data.sample_mating_pairs(75, 123).unwrap();
//         assert_eq!(a.len(), 75);
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn sample_mating_pairs_respects_dioecious_sexes() {
//         let ctx = context();
//         let data = Data::new(&ctx, 100, 5, 100, 1, 2, true, 42).unwrap();
//         let pairs = data.sample_mating_pairs(200, 123).unwrap();
//         for (p1, p2) in pairs {
//             assert_eq!(data.sexes[p1], Sex::Homogametic);
//             assert_eq!(data.sexes[p2], Sex::Heterogametic);
//         }
//     }

//     #[test]
//     fn mate_generates_correct_offspring_names() {
//         let ctx = context();
//         let parent = Data::new(&ctx, 10, 2, 5, 2, 2, true, 42).unwrap();
//         let offspring = parent.mate(vec![(2, 7), (0, 9)], &ctx, 111).unwrap();
//         assert_eq!(
//             offspring.entries[0].name,
//             format!("{}--x--{}", parent.entries[2].name, parent.entries[7].name)
//         );
//         assert_eq!(
//             offspring.entries[1].name,
//             format!("{}--x--{}", parent.entries[0].name, parent.entries[9].name)
//         );
//     }

//     #[test]
//     fn mate_preserves_haplotype_persistence_probabilities() {
//         let ctx = context();
//         let parent = Data::new(&ctx, 20, 3, 40, 2, 2, true, 42).unwrap();
//         let pairs = parent.sample_mating_pairs(5, 123).unwrap();
//         let offspring = parent.mate(pairs, &ctx, 456).unwrap();
//         assert_eq!(
//             parent.haplotype_persistence_probs,
//             offspring.haplotype_persistence_probs
//         );
//     }

//     #[test]
//     fn mate_produces_deterministic_results() {
//         let ctx = context();
//         let parent = Data::new(&ctx, 10, 2, 20, 2, 2, true, 42).unwrap();
//         let pairs = parent.sample_mating_pairs(5, 123).unwrap();
//         let a = parent.mate(pairs.clone(), &ctx, 999).unwrap();
//         let b = parent.mate(pairs, &ctx, 999).unwrap();
//         assert_eq!(
//             a.genotype_data.to_vec_f32(&ctx).unwrap(),
//             b.genotype_data.to_vec_f32(&ctx).unwrap()
//         );
//     }

//     #[test]
//     fn mate_high_haplotype_persistence_prevents_crossovers() {
//         let ctx = context();
//         let mut parent_data = Data::new(&ctx, 10, 3, 30, 2, 2, false, 42).unwrap();
//         let _ = &mut parent_data.haplotype_persistence_probs.fill(1.0);
//         let pairs = vec![(0, 1), (2, 3), (4, 5)];
//         let offspring_data = parent_data.mate(pairs.clone(), &ctx, 123).unwrap();
//         let parent_vec = parent_data.genotype_data.to_vec_f32(&ctx).unwrap();
//         let offspring_vec = offspring_data.genotype_data.to_vec_f32(&ctx).unwrap();
//         let n_loci_alleles = parent_data.genotype_data.shape[1] as usize;
//         for (off_idx, &(p1_idx, p2_idx)) in pairs.iter().enumerate() {
//             for chr_idx in 0..parent_data.genome.len() {
//                 let mut chr_cols: Vec<usize> = Vec::new();
//                 for locus in &parent_data.loci {
//                     if locus.chromosome_id == chr_idx {
//                         chr_cols.extend(&locus.col_idx);
//                     }
//                 }
//                 let p1_base = p1_idx * n_loci_alleles * 2;
//                 let p2_base = p2_idx * n_loci_alleles * 2;
//                 let off_base = off_idx * n_loci_alleles * 2;
//                 let mut p1_h0 = Vec::new();
//                 let mut p1_h1 = Vec::new();
//                 let mut p2_h0 = Vec::new();
//                 let mut p2_h1 = Vec::new();
//                 let mut off_h0 = Vec::new();
//                 let mut off_h1 = Vec::new();
//                 for &col in &chr_cols {
//                     p1_h0.push(parent_vec[p1_base + (2 * col)]);
//                     p1_h1.push(parent_vec[p1_base + (2 * col) + 1]);
//                     p2_h0.push(parent_vec[p2_base + (2 * col)]);
//                     p2_h1.push(parent_vec[p2_base + (2 * col) + 1]);
//                     off_h0.push(offspring_vec[off_base + (2 * col)]);
//                     off_h1.push(offspring_vec[off_base + (2 * col) + 1]);
//                 }
//                 assert!(
//                     off_h0 == p1_h0 || off_h0 == p1_h1,
//                     "Offspring {} inherited a recombinant haplotype from parent {} on chromosome {} despite complete haplotype persistence.",
//                     off_idx,
//                     p1_idx,
//                     chr_idx
//                 );
//                 assert!(
//                     off_h1 == p2_h0 || off_h1 == p2_h1,
//                     "Offspring {} inherited a recombinant haplotype from parent {} on chromosome {} despite complete haplotype persistence.",
//                     off_idx,
//                     p2_idx,
//                     chr_idx
//                 );
//             }
//         }
//     }

//     #[test]
//     fn sim_allele_effects_rejects_invalid_parameters() {
//         let ctx = context();
//         let mut data = Data::new(&ctx, 10, 2, 20, 2, 2, false, 42).unwrap();
//         assert!(
//             data.sim_allele_effects(&ctx, &[(5, 0.0, 1.0)], 123)
//                 .is_err()
//         );
//         assert!(
//             data.sim_allele_effects(&ctx, &[(21, 0.0, 1.0), (5, 0.0, 1.0)], 123)
//                 .is_err()
//         );
//         assert!(
//             data.sim_allele_effects(&ctx, &[(5, 0.0, -1.0), (5, 0.0, 1.0)], 123)
//                 .is_err()
//         );
//     }

//     #[test]
//     fn phenotypes_rejects_invalid_parameter() {
//         let ctx = context();
//         let mut data = Data::new(&ctx, 10, 2, 20, 2, 2, false, 42).unwrap();
//         let effects = data
//             .sim_allele_effects(&ctx, &[(5, 0.0, 1.0), (5, 0.0, 1.0)], 123)
//             .unwrap();
//         assert!(data.phenotype(&effects, &ctx, &[1.0], 123).is_err());
//         assert!(
//             data.phenotype(&effects, &ctx, &[-1.0, 1.0], 123)
//                 .is_err()
//         );
//     }

//     #[test]
//     fn phenotypes_is_deterministic_and_updates_dat() {
//         let ctx = context();
//         let mut a = Data::new(&ctx, 10, 2, 20, 2, 2, false, 42).unwrap();
//         let mut b = Data::new(&ctx, 10, 2, 20, 2, 2, false, 42).unwrap();
//         let e1 = a
//             .sim_allele_effects(&ctx, &[(5, 0.0, 1.0), (5, 0.0, 1.0)], 123)
//             .unwrap();
//         let e2 = b
//             .sim_allele_effects(&ctx, &[(5, 0.0, 1.0), (5, 0.0, 1.0)], 123)
//             .unwrap();
//         a.phenotype(&e1, &ctx, &[1.0, 1.0], 456).unwrap();
//         b.phenotype(&e2, &ctx, &[1.0, 1.0], 456).unwrap();
//         assert_eq!(
//             a.phenotype_data.to_vec_f32(&ctx).unwrap(),
//             b.phenotype_data.to_vec_f32(&ctx).unwrap()
//         );
//     }

//     #[test]
//     fn phenotypes_are_approximately_normally_distribute() {
//         let ctx = context();
//         // Use a large population and sufficient loci so the Central Limit Theorem kicks in
//         let n_entries = 5000;
//         let n_loci = 200;
//         let n_traits = 1;
//         let mut data = Data::new(&ctx, n_entries, 5, n_loci, n_traits, 2, false, 42).unwrap();
//         // 100 loci with effects, mean=0.0, genetic sd=1.0
//         let genetic_params = [(100, 0.0, 1.0)];
//         let effects = data.sim_allele_effects(&ctx, &genetic_params, 123).unwrap();
//         // env error sd=1.0
//         data.phenotype(&effects, &ctx, &[1.0], 456).unwrap();
//         let phenotypes = data.phenotype_data.to_vec_f32(&ctx).unwrap();
//         assert_eq!(phenotypes.len(), n_entries);
//         // 1. Calculate Mean
//         let mean: f32 = phenotypes.iter().sum::<f32>() / (n_entries as f32);
//         // 2. Calculate Variance and Standard Deviation
//         let variance: f32 =
//             phenotypes.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / ((n_entries - 1) as f32);
//         let sd = variance.sqrt();
//         // 3. Check Empirical Rule (68-95-99.7)
//         let mut within_1_sd = 0;
//         let mut within_2_sd = 0;
//         let mut within_3_sd = 0;
//         for &p in &phenotypes {
//             let diff = (p - mean).abs();
//             if diff <= sd {
//                 within_1_sd += 1;
//             }
//             if diff <= 2.0 * sd {
//                 within_2_sd += 1;
//             }
//             if diff <= 3.0 * sd {
//                 within_3_sd += 1;
//             }
//         }
//         let prop_1_sd = (within_1_sd as f32) / (n_entries as f32);
//         let prop_2_sd = (within_2_sd as f32) / (n_entries as f32);
//         let prop_3_sd = (within_3_sd as f32) / (n_entries as f32);
//         // Allow a +/- 4% tolerance for randomness
//         assert!(
//             (prop_1_sd - 0.6827).abs() < 0.04,
//             "Expected ~68.3% within 1 SD, got {:.2}%",
//             prop_1_sd * 100.0
//         );
//         assert!(
//             (prop_2_sd - 0.9545).abs() < 0.04,
//             "Expected ~95.5% within 2 SD, got {:.2}%",
//             prop_2_sd * 100.0
//         );
//         assert!(
//             (prop_3_sd - 0.9973).abs() < 0.02, // 3 SD is usually very close to 100%
//             "Expected ~99.7% within 3 SD, got {:.2}%",
//             prop_3_sd * 100.0
//         );
//         // 4. Check Skewness (should be near 0 for a normal distribution)
//         let skewness: f32 = phenotypes.iter().map(|&x| (x - mean).powi(3)).sum::<f32>()
//             / ((n_entries as f32) * sd.powi(3));
//         assert!(
//             skewness.abs() < 0.2,
//             "Expected symmetrical distribution (skewness near 0), got {}",
//             skewness
//         );
//     }
// }
