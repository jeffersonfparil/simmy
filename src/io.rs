use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use rand::prelude::IndexedRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp, Normal};
use std::borrow::Cow;
use std::fmt;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub name: String,
    pub lengths: (usize, usize),  // homologous chromosome lengths
    pub ld_decay_distance: usize, // will be used in mating assuming r(d) = exp(-d/L), where d is the distance between a pair of loci in bases and L is ld_decay_distance.
    pub is_sex_chromosome: bool,
}

#[derive(Debug, Clone)]
pub struct Locus {
    pub chromosome_id: usize, // index of the chromosome containing this locus, which assumes one or more chromosomes are stored in a vector (contiguous/ordered list)
    pub position: usize,      // position in the chromosome
    pub alleles: Vec<String>, // sequence of each allele
    pub length: usize, // maximum size of alleles, i.e. the number of bases of the longest allele
    pub col_idx: Vec<usize>, // The column indices in the main genotype tensor, each referring to an allele
}

#[derive(Debug, Clone)]
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
    pub traits: Vec<Trait>,
    pub genotype_data: GpuTensor, // 3D tensor with shape: n_entries x n_loci_alleles x maternal+paternal haplotypes
    pub phenotype_data: GpuTensor, // 2D tensor with shape: n_entries x n_traits (additionally monogametic = 0.0 and heterogametic = 1.0)
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
struct MeiosisParams {
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
struct MatingPairData {
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
struct LocusData {
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

impl Data {
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
    ) -> Result<Self> {
        // This is intended as a generic not totally biologically realistic initialiser for the Data struct,
        // where future methods will mutate the resulting struct with more biologically realistic information.
        ensure!(
            n_entries > 0,
            "The number of entries need to be greater than zero!"
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
            n_traits > 0,
            "The number of traits need to be greater than zero!"
        );
        ensure!(ploidy > 0, "Ploidy need to be greater than zero!");
        ensure!(
            ploidy.is_multiple_of(2),
            "We do not support odd ploidies at the moment!"
        );
        // Entries
        let mut entries: Vec<Entry> = Vec::with_capacity(n_entries);
        let n_digits: usize = format!("{}", n_entries).len();
        for i in 0..n_entries {
            entries.push(Entry {
                name: format!("entry_{:0>n_digits$}", i),
                species: "".to_owned(),
                group: "".to_owned(),
                notes: "".to_owned(),
            });
        }
        // Genome
        let mut genome: Vec<Chromosome> = Vec::with_capacity(n_chromosomes);
        let n_digits: usize = format!("{}", n_chromosomes).len();
        for i in 0..n_chromosomes {
            genome.push(Chromosome {
                name: format!("chromosome_{:0>n_digits$}", i),
                lengths: (1_000_000, 1_000_000),
                ld_decay_distance: 10_000,
                is_sex_chromosome: with_sex && (i == (n_chromosomes - 1)),
            });
        }
        // Loci
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
            let n = chromosome.lengths.0;
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
        // Traits
        let mut traits: Vec<Trait> = Vec::with_capacity(n_traits);
        let n_digits: usize = format!("{}", n_traits).len();
        for i in 0..n_traits {
            traits.push(Trait {
                name: format!("trait_{:0>n_digits$}", i),
                description: "".to_owned(),
            });
        }
        // Sexes
        let mut sexes: Vec<Sex> = Vec::with_capacity(n_entries);
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
        // Genotype data
        // Note that we are not simulating LD between loci pairs yet, i.e. assuming the resulting alleles are historical effects
        let mut genotype_data_tmp: Vec<f32> = vec![0.0; n_entries * n_loci_alleles * 2];
        let beta_n =
            Beta::new(2.0, 5.0).expect("Failed to initialise a Beta distribution (a=2; b=5)!"); // We will sample allele indices per locus from a skewed distribution, i.e. biased towards the first allele which is more realistic than uniform distribution.
        let beta_u =
            Beta::new(0.5, 0.5).expect("Failed to initialise a Beta distribution (a=b=0.5)!"); // We will sample from a realistic U shaped distribution for genomewide loci on a single population.
        for (i, &sex) in sexes.iter().enumerate() {
            let idx_entry = i * 2 * n_loci_alleles;
            for locus in &loci {
                if !genome[locus.chromosome_id].is_sex_chromosome {
                    let n_alleles = locus.col_idx.len();
                    // Sample alleles and their dosages per homologous chromosome or parent,
                    // where the same allele may be sampled and hence fixed on a homologous chromosome and so we add to the initialised 0.0 dosages.
                    for j in 0..2 {
                        // Each allele from each ploidy is sampled here
                        for _ in 0..(ploidy / 2) {
                            let idx_allele_1: usize = locus.col_idx[(beta_n.sample(&mut rng)
                                * ((n_alleles - 1) as f32))
                                .round()
                                as usize];
                            let idx_allele_2: usize = locus.col_idx[(beta_n.sample(&mut rng)
                                * ((n_alleles - 1) as f32))
                                .round()
                                as usize];
                            let allele_1_dosage_parent_j = (beta_u.sample(&mut rng) as f32).round();
                            let allele_2_dosage_parent_j = 1.00 - allele_1_dosage_parent_j;
                            genotype_data_tmp[idx_entry + (2 * idx_allele_1) + j] +=
                                allele_1_dosage_parent_j;
                            genotype_data_tmp[idx_entry + (2 * idx_allele_2) + j] +=
                                allele_2_dosage_parent_j;
                        }
                    }
                } else {
                    let idx_allele_1: usize = locus.col_idx[0];
                    let idx_allele_2: usize = locus.col_idx[1];
                    match sex {
                        Sex::Homogametic => {
                            genotype_data_tmp[idx_entry + (2 * idx_allele_1)] +=
                                (ploidy / 2) as f32;
                            genotype_data_tmp[idx_entry + (2 * idx_allele_1) + 1] += 0.0;
                            genotype_data_tmp[idx_entry + (2 * idx_allele_2)] +=
                                (ploidy / 2) as f32;
                            genotype_data_tmp[idx_entry + (2 * idx_allele_2) + 1] += 0.0;
                        }
                        Sex::Heterogametic => {
                            genotype_data_tmp[idx_entry + (2 * idx_allele_1)] +=
                                (ploidy / 2) as f32;
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
        let genotype_data = GpuTensor::from_f32(
            ctx,
            &genotype_data_tmp,
            &[n_entries as u32, n_loci_alleles as u32, 2],
            None,
            None,
        )?;
        // Phenotype data
        // Note that we are not linking the phenotype with the genotype data in this initialisation method
        // Similarly, sex-related phenotypes are not defined regardless of the presense of sex chromosomes and homo- and heterogametic entries
        let mut phenotype_data_tmp: Vec<f32> = vec![0.0; n_entries * n_traits];
        let normal =
            Normal::new(0.0, 1.0).expect("Failed to initialise a standard normal distribution!");
        for i in 0..n_entries {
            for j in 0..n_traits {
                phenotype_data_tmp[(i * n_traits) + j] = normal.sample(&mut rng);
            }
        }
        let phenotype_data = GpuTensor::from_f32(
            ctx,
            &phenotype_data_tmp,
            &[n_entries as u32, n_traits as u32],
            None,
            None,
        )?;
        // Output
        Ok(Self {
            entries,
            ploidy,
            sexes,
            genome,
            loci,
            traits,
            genotype_data,
            phenotype_data,
        })
    }
    pub fn check_dimensions(&self) -> Result<()> {
        let n_entries: usize = self.entries.len();
        let n_loci: usize = self.loci.len();
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
    pub fn prob_linkage(&self, idx_locus_1: usize, idx_locus_2: usize) -> Result<f64> {
        self.check_dimensions()?;
        let r: f64 = if self.loci[idx_locus_1].chromosome_id == self.loci[idx_locus_2].chromosome_id
        {
            let idx_chromosome: usize = self.loci[idx_locus_1].chromosome_id;
            let ld_decay_distance: f64 = self.genome[idx_chromosome].ld_decay_distance as f64;
            let distance: f64 = {
                let position_1: usize = self.loci[idx_locus_1].position;
                let position_2: usize = self.loci[idx_locus_2].position;
                (position_1 as f64 - position_2 as f64).abs()
            };
            (-distance / ld_decay_distance).exp().max(0.5) // ranges from 0.5 (no linkage) to 1.0 (complete linkage)
        } else {
            0.5
        };
        Ok(r)
    }
    pub fn mate(
        &self,
        mating_pairs: Vec<(usize, usize)>,
        ctx: &GpuContext,
        seed: u64,
    ) -> Result<Self> {
        self.check_dimensions()?;
        let n_offsprings = mating_pairs.len();
        let n_loci = self.loci.len();
        let n_loci_alleles = self.loci.iter().fold(0, |sum, x| sum + x.col_idx.len());
        // Locus Data
        let mut locus_data_packed = Vec::with_capacity(n_loci);
        for j in 0..n_loci {
            let start_col = *self.loci[j].col_idx.first().unwrap() as u32;
            let end_col = (*self.loci[j].col_idx.last().unwrap() as u32) + 1;
            let r = if j == 0 {
                0.5
            } else {
                self.prob_linkage(j - 1, j)? as f32
            };
            let mut metadata = 0u32;
            let chr_id = self.loci[j].chromosome_id;
            if self.genome[chr_id].is_sex_chromosome {
                metadata |= 1;
            }
            if j == 0 || chr_id != self.loci[j - 1].chromosome_id {
                metadata |= 2;
            }
            locus_data_packed.push(LocusData {
                start_col,
                end_col,
                r,
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
            pairs_data_packed.push(MatingPairData {
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
        let params = MeiosisParams {
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
        let mut offsprings = Data::new(
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
        Ok(offsprings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::context::GpuContext;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    #[test]
    fn rejects_zero_entries() {
        let ctx = context();
        assert!(Data::new(&ctx, 0, 1, 1, 1, 2, false, 42).is_err());
    }
    #[test]
    fn rejects_zero_chromosomes() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 0, 1, 1, 2, false, 42).is_err());
    }
    #[test]
    fn rejects_zero_loci() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 0, 1, 2, false, 42).is_err());
    }
    #[test]
    fn rejects_zero_traits() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 0, 2, false, 42).is_err());
    }
    #[test]
    fn rejects_zero_ploidy() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 1, 0, false, 42).is_err());
    }
    #[test]
    fn rejects_odd_ploidy() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 1, 3, false, 42).is_err());
    }
    #[test]
    fn rejects_fewer_loci_than_chromosomes() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 10, 5, 1, 2, false, 42).is_err());
    }
    #[test]
    fn creates_expected_counts() {
        let ctx = context();
        let data = Data::new(&ctx, 25, 5, 100, 7, 2, false, 42).unwrap();
        println!("data: {}", data);
        assert_eq!(data.entries.len(), 25);
        assert_eq!(data.sexes.len(), 25);
        assert_eq!(data.genome.len(), 5);
        assert_eq!(data.loci.len(), 100);
        assert_eq!(data.traits.len(), 7);
    }
    #[test]
    fn locus_count_equals_requested() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 7, 103, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        assert_eq!(data.loci.len(), 103);
    }
    #[test]
    fn every_chromosome_receives_loci() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 10, 10, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        let mut counts = [0usize; 10];
        for locus in &data.loci {
            counts[locus.chromosome_id] += 1;
        }
        assert!(counts.iter().all(|&n| n > 0));
    }
    #[test]
    fn chromosome_positions_are_sorted() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        for chr in 0..data.genome.len() {
            let positions: Vec<_> = data
                .loci
                .iter()
                .filter(|l| l.chromosome_id == chr)
                .map(|l| l.position)
                .collect();
            assert!(positions.windows(2).all(|w| w[0] <= w[1]));
        }
    }
    #[test]
    fn locus_column_indices_are_contiguous_and_unique() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        let mut all: Vec<usize> = data
            .loci
            .iter()
            .flat_map(|l| l.col_idx.iter().copied())
            .collect();
        all.sort_unstable();
        for (i, idx) in all.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }
    #[test]
    fn allele_count_matches_column_count() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        for locus in &data.loci {
            assert_eq!(locus.alleles.len(), locus.col_idx.len());
        }
    }
    #[test]
    fn allele_count_is_valid() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        for locus in &data.loci {
            assert!((2..=5).contains(&locus.alleles.len()));
        }
    }
    #[test]
    fn all_entries_are_hermaphrodites_without_sex() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 1, 2, false, 42).unwrap();
        println!("data: {}", data);
        assert!(data.sexes.iter().all(|s| *s == Sex::Hermaphrodite));
    }
    #[test]
    fn no_hermaphrodites_with_sex_enabled() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 1, 2, true, 42).unwrap();
        println!("data: {}", data);
        assert!(data.sexes.iter().all(|s| *s != Sex::Hermaphrodite));
    }
    #[test]
    fn sex_ratio_is_approximately_fifty_fifty() {
        let ctx = context();
        let data = Data::new(&ctx, 1000, 5, 100, 1, 2, true, 42).unwrap();
        println!("data: {}", data);
        let homo = data
            .sexes
            .iter()
            .filter(|&&s| s == Sex::Homogametic)
            .count();
        let hetero = data
            .sexes
            .iter()
            .filter(|&&s| s == Sex::Heterogametic)
            .count();
        let frac_homo = homo as f64 / (homo + hetero) as f64;
        assert!((frac_homo - 0.5).abs() < 0.10);
    }
    #[test]
    fn check_dimensions_passes() {
        let ctx = context();
        let data = Data::new(&ctx, 10, 5, 100, 3, 4, true, 42).unwrap();
        println!("data: {}", data);
        assert!(data.check_dimensions().is_ok());
    }
    #[test]
    fn total_locus_dosage_equals_ploidy() {
        let ctx = context();
        for ploidy in [2usize, 4, 6, 8, 10] {
            let data = Data::new(&ctx, 20, 5, 100, 1, ploidy, true, 42).unwrap();
            println!("data: {}", data);
            let genotype = data.genotype_data.to_vec_f32(&ctx).unwrap();
            let n_loci_alleles: usize = data.loci.iter().map(|l| l.col_idx.len()).sum();
            for entry_idx in 0..data.entries.len() {
                let base = entry_idx * n_loci_alleles * 2;
                for locus in &data.loci {
                    let total_dosage: usize = locus
                        .col_idx
                        .iter()
                        .map(|&col| {
                            (genotype[base + (2 * col)] + genotype[base + (2 * col) + 1]) as usize
                        })
                        .sum();
                    assert_eq!(
                        total_dosage, ploidy,
                        "entry={}, chromosome={}, position={}, ploidy={}",
                        entry_idx, locus.chromosome_id, locus.position, ploidy
                    );
                }
            }
        }
    }
    #[test]
    fn sample_mating_pairs_returns_correct_number_and_is_deterministic() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 1, 2, true, 42).unwrap();
        println!("data: {}", data);
        let n_offspring = 75;
        let pairs_run_1 = data.sample_mating_pairs(n_offspring, 123).unwrap();
        let pairs_run_2 = data.sample_mating_pairs(n_offspring, 123).unwrap();
        assert_eq!(pairs_run_1.len(), n_offspring);
        assert_eq!(
            pairs_run_1, pairs_run_2,
            "Sampling with the same seed must produce identical pairs."
        );
    }
    #[test]
    fn sample_mating_pairs_respects_dioecious_sexes() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 1, 2, true, 42).unwrap();
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(200, 123).unwrap();
        for (parent_1, parent_2) in pairs {
            assert_eq!(
                data.sexes[parent_1],
                Sex::Homogametic,
                "First parent must be Homogametic"
            );
            assert_eq!(
                data.sexes[parent_2],
                Sex::Heterogametic,
                "Second parent must be Heterogametic"
            );
        }
    }
    #[test]
    fn sample_mating_pairs_works_with_hermaphrodites() {
        let ctx = context();
        let data = Data::new(&ctx, 50, 5, 100, 1, 2, false, 42).unwrap(); // with_sex = false sets all to Hermaphrodite
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(100, 123).unwrap();
        for (parent_1, parent_2) in pairs {
            assert_eq!(data.sexes[parent_1], Sex::Hermaphrodite);
            assert_eq!(data.sexes[parent_2], Sex::Hermaphrodite);
        }
    }
    #[test]
    fn sample_mating_pairs_fails_if_homogametics_are_missing() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 5, 10, 1, 2, true, 42).unwrap();
        // Manually mutate the population to strictly Heterogametic
        for sex in data.sexes.iter_mut() {
            *sex = Sex::Heterogametic;
        }
        println!("data: {}", data);
        let result = data.sample_mating_pairs(5, 123);
        assert!(
            result.is_err(),
            "Mating should fail if there are no Homogametic or Hermaphrodite entries."
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "There should be hermaphroditic and/or homogametic entries!"
        );
    }
    #[test]
    fn sample_mating_pairs_fails_if_heterogametics_are_missing() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 5, 10, 1, 2, true, 42).unwrap();
        // Manually mutate the population to strictly Homogametic
        for sex in data.sexes.iter_mut() {
            *sex = Sex::Homogametic;
        }
        println!("data: {}", data);
        let result = data.sample_mating_pairs(5, 123);
        assert!(
            result.is_err(),
            "Mating should fail if there are no Heterogametic or Hermaphrodite entries."
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "There should be hermaphroditic and/or heterogametic entries!"
        );
    }
    #[test]
    fn prob_linkage_different_chromosomes_returns_half() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 2, 10, 1, 2, false, 42).unwrap();
        // Force locus 0 to chromosome 0 and locus 1 to chromosome 1
        data.loci[0].chromosome_id = 0;
        data.loci[1].chromosome_id = 1;
        let r = data.prob_linkage(0, 1).unwrap();
        assert_eq!(
            r, 0.5,
            "Loci on different chromosomes must assort independently (r=0.5)"
        );
    }
    #[test]
    fn prob_linkage_same_chromosome_zero_distance_returns_one() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 1, 10, 1, 2, false, 42).unwrap();
        // Force loci to the exact same position on the same chromosome
        data.loci[0].chromosome_id = 0;
        data.loci[0].position = 1000;
        data.loci[1].chromosome_id = 0;
        data.loci[1].position = 1000;
        let r = data.prob_linkage(0, 1).unwrap();
        assert_eq!(
            r, 1.0,
            "Loci at the exact same position must have complete linkage (r=1.0)"
        );
    }
    #[test]
    fn prob_linkage_same_chromosome_intermediate_distance() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 1, 10, 1, 2, false, 42).unwrap();
        // Set a clean decay distance for predictable math
        data.genome[0].ld_decay_distance = 10_000;
        data.loci[0].chromosome_id = 0;
        data.loci[0].position = 10_000;
        data.loci[1].chromosome_id = 0;
        data.loci[1].position = 12_231; // distance = 2231
        // Mathematical expectation:
        // distance / ld_decay = 2231 / 10000 = 0.2231
        // r = exp(-0.2231) ≈ 0.8
        let r = data.prob_linkage(0, 1).unwrap();
        let expected = (-0.2231f64).exp();
        assert!(
            (r - expected).abs() < 1e-6,
            "Expected intermediate linkage {}, got {}",
            expected,
            r
        );
    }
    #[test]
    fn prob_linkage_large_distance_capped_at_half() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 1, 10, 1, 2, false, 42).unwrap();
        data.genome[0].ld_decay_distance = 10_000;
        data.loci[0].chromosome_id = 0;
        data.loci[0].position = 0;
        data.loci[1].chromosome_id = 0;
        data.loci[1].position = 100_000; // distance = 100,000
        // exp(-100,000 / 10,000) = exp(-10) ≈ 0.000045
        // This must be bounded to 0.5 by .max(0.5)
        let r = data.prob_linkage(0, 1).unwrap();
        assert_eq!(
            r, 0.5,
            "Loci separated by vast distances should be capped at independent assortment (r=0.5)"
        );
    }
    #[test]
    fn mate_creates_correct_dimensions() {
        let ctx = context();
        let parent_data = Data::new(&ctx, 10, 2, 20, 2, 2, true, 42).unwrap();
        println!("parent_data: {}", parent_data);
        // Generate 15 offspring from random mating pairs
        let pairs = parent_data.sample_mating_pairs(15, 123).unwrap();
        let offspring_data = parent_data.mate(pairs, &ctx, 456).unwrap();
        println!("offspring_data: {}", offspring_data);
        assert_eq!(offspring_data.entries.len(), 15);
        assert_eq!(offspring_data.genotype_data.shape[0] as usize, 15); // 15 offspring
        assert_eq!(
            offspring_data.genotype_data.shape[1],
            parent_data.genotype_data.shape[1]
        ); // Loci-alleles preserved
        assert_eq!(offspring_data.genotype_data.shape[2], 2); // 2 homologous chromosomes
    }
    #[test]
    fn mate_produces_deterministic_results() {
        let ctx = context();
        let parent_data = Data::new(&ctx, 10, 2, 20, 2, 2, true, 42).unwrap();
        println!("parent_data: {}", parent_data);
        let pairs = parent_data.sample_mating_pairs(5, 123).unwrap();
        // Mate twice with the exact same seed
        let offspring_1 = parent_data.mate(pairs.clone(), &ctx, 999).unwrap();
        let offspring_2 = parent_data.mate(pairs, &ctx, 999).unwrap();
        // Download the GPU buffers
        let vec_1 = offspring_1.genotype_data.to_vec_f32(&ctx).unwrap();
        let vec_2 = offspring_2.genotype_data.to_vec_f32(&ctx).unwrap();
        assert_eq!(
            vec_1, vec_2,
            "GPU Compute Shader must produce perfectly deterministic genotypes for the same seed."
        );
    }
    #[test]
    fn mate_generates_correct_offspring_names() {
        let ctx = context();
        let parent_data = Data::new(&ctx, 10, 2, 5, 2, 2, true, 42).unwrap();
        println!("parent_data: {}", parent_data);
        // Manually assign pairs to guarantee exact indices
        let pairs = vec![(2, 7), (0, 9)];
        let p2_name = &parent_data.entries[2].name;
        let p7_name = &parent_data.entries[7].name;
        let p0_name = &parent_data.entries[0].name;
        let p9_name = &parent_data.entries[9].name;
        let offspring_data = parent_data.mate(pairs, &ctx, 111).unwrap();
        println!("offspring_data: {}", offspring_data);
        assert_eq!(
            offspring_data.entries[0].name,
            format!("{}--x--{}", p2_name, p7_name)
        );
        assert_eq!(
            offspring_data.entries[1].name,
            format!("{}--x--{}", p0_name, p9_name)
        );
    }
    #[test]
    fn mate_preserves_total_ploidy_per_autosomal_locus() {
        let ctx = context();
        let ploidy = 2; // Diploid
        let parent_data = Data::new(&ctx, 10, 3, 15, 2, ploidy, true, 42).unwrap();
        println!("parent_data: {}", parent_data);
        let pairs = parent_data.sample_mating_pairs(5, 123).unwrap();
        let offspring_data = parent_data.mate(pairs, &ctx, 777).unwrap();
        println!("offspring_data: {}", offspring_data);
        // Download the genotype tensor to evaluate the allele sums
        let genotype = offspring_data.genotype_data.to_vec_f32(&ctx).unwrap();
        let n_loci_alleles = offspring_data.genotype_data.shape[1] as usize;
        for entry_idx in 0..offspring_data.entries.len() {
            let base = entry_idx * n_loci_alleles * 2;
            for locus in &offspring_data.loci {
                // Skip sex chromosomes for this strict check, as their dosage
                // varies based on homogametic vs heterogametic sex inheritance
                if offspring_data.genome[locus.chromosome_id].is_sex_chromosome {
                    continue;
                }
                // Sum all allele dosages for this specific locus across both homologous chromosomes
                let total_dosage: f32 = locus
                    .col_idx
                    .iter()
                    .map(|&col| genotype[base + (2 * col)] + genotype[base + (2 * col) + 1])
                    .sum();
                // Due to floating point math inside f32, we check with a small epsilon
                assert!(
                    (total_dosage - ploidy as f32).abs() < 1e-4,
                    "Autosomal locus dosage {} does not equal expected ploidy {} at offspring {}, locus {}",
                    total_dosage,
                    ploidy,
                    entry_idx,
                    locus.position
                );
            }
        }
    }
    #[test]
    fn mate_high_linkage_prevents_crossovers() {
        let ctx = context();
        // Create parent population (using false for sex to simplify to hermaphrodites for the test)
        let mut parent_data = Data::new(&ctx, 10, 3, 30, 2, 2, false, 42).unwrap();
        // Force extremely high linkage on all chromosomes to completely suppress crossovers
        for chr in &mut parent_data.genome {
            // A massive LD decay distance ensures `r` evaluates to 1.0 (complete linkage)
            chr.ld_decay_distance = 1_000_000_000_000;
        }
        // Mate specific pairs
        let pairs = vec![(0, 1), (2, 3), (4, 5)];
        let offspring_data = parent_data.mate(pairs.clone(), &ctx, 123).unwrap();
        println!("offspring_data: {}", offspring_data);
        // Download both tensors to host memory for comparison
        let parent_vec = parent_data.genotype_data.to_vec_f32(&ctx).unwrap();
        let offspring_vec = offspring_data.genotype_data.to_vec_f32(&ctx).unwrap();
        let n_loci_alleles = parent_data.genotype_data.shape[1] as usize;
        for (off_idx, &(p1_idx, p2_idx)) in pairs.iter().enumerate() {
            for chr_idx in 0..parent_data.genome.len() {
                // Find all locus allele columns belonging to this specific chromosome
                let mut chr_cols: Vec<usize> = Vec::new();
                for locus in &parent_data.loci {
                    if locus.chromosome_id == chr_idx {
                        chr_cols.extend(&locus.col_idx);
                    }
                }
                let mut p1_homolog_0 = Vec::new();
                let mut p1_homolog_1 = Vec::new();
                let mut p2_homolog_0 = Vec::new();
                let mut p2_homolog_1 = Vec::new();
                let mut off_homolog_0 = Vec::new();
                let mut off_homolog_1 = Vec::new();
                let p1_base = p1_idx * n_loci_alleles * 2;
                let p2_base = p2_idx * n_loci_alleles * 2;
                let off_base = off_idx * n_loci_alleles * 2;
                // Extract the haplotypes for this entire chromosome
                for &col in &chr_cols {
                    p1_homolog_0.push(parent_vec[p1_base + (2 * col)]);
                    p1_homolog_1.push(parent_vec[p1_base + (2 * col) + 1]);
                    p2_homolog_0.push(parent_vec[p2_base + (2 * col)]);
                    p2_homolog_1.push(parent_vec[p2_base + (2 * col) + 1]);
                    off_homolog_0.push(offspring_vec[off_base + (2 * col)]);
                    off_homolog_1.push(offspring_vec[off_base + (2 * col) + 1]);
                }
                // Since recombination was completely suppressed, the offspring's first
                // homologous chromosome MUST be a perfect copy of one of Parent 1's homologs
                let matches_p1_h0 = off_homolog_0 == p1_homolog_0;
                let matches_p1_h1 = off_homolog_0 == p1_homolog_1;
                assert!(
                    matches_p1_h0 || matches_p1_h1,
                    "Offspring {} experienced a crossover on chromosome {} from Parent {} despite high linkage!",
                    off_idx,
                    chr_idx,
                    p1_idx
                );
                // Similarly for Parent 2 and the offspring's second homologous chromosome
                let matches_p2_h0 = off_homolog_1 == p2_homolog_0;
                let matches_p2_h1 = off_homolog_1 == p2_homolog_1;
                assert!(
                    matches_p2_h0 || matches_p2_h1,
                    "Offspring {} experienced a crossover on chromosome {} from Parent {} despite high linkage!",
                    off_idx,
                    chr_idx,
                    p2_idx
                );
            }
        }
    }
}
