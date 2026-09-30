use crate::entry::*;
use crate::genotype::*;
use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use crate::phenotype::*;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use rand::prelude::IndexedRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use std::borrow::Cow;
use std::fmt;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Dataset {
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

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "------------------------")?;
        writeln!(f, "Dataset struct")?;
        writeln!(f, "------------------------")?;
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
        writeln!(
            f,
            "\t- Locus Alleles: {}",
            self.loci.iter().map(|l| l.col_idx.len()).sum::<usize>()
        )?;
        writeln!(f, "\t- Haplotype persistence probability")?;
        writeln!(
            f,
            "\t  (probabilities of adjacent loci pairs being inherited together): {}",
            self.haplotype_persistence_probs.len()
        )?;
        writeln!(f, "\t- Traits: {}", self.traits.len())?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Genotype Tensor: {}", self.genotype_data)?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Phenotype Tensor: {}", self.phenotype_data)?;
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

impl Dataset {
    pub fn new(ctx: &GpuContext, n: usize) -> Result<Self> {
        let entries: Vec<Entry> = vec![
            Entry {
                name: "".to_owned(),
                species: "".to_owned(),
                group: "".to_owned(),
                notes: "".to_owned()
            };
            n
        ];
        let sexes: Vec<Sex> = vec![Sex::Hermaphrodite; n];
        let genome: Vec<Chromosome> = Vec::with_capacity(10);
        let loci: Vec<Locus> = Vec::with_capacity(1);
        let haplotype_persistence_probs: Vec<f32> = Vec::with_capacity(1);
        let traits: Vec<Trait> = Vec::with_capacity(1);
        let genotype_data: GpuTensor = GpuTensor::from_vec_f32(ctx, &[0.0], &[1], None, None)?;
        let phenotype_data: GpuTensor = GpuTensor::from_vec_f32(ctx, &[0.0], &[1], None, None)?;
        Ok(Self {
            entries,
            ploidy: 2,
            sexes,
            genome,
            loci,
            haplotype_persistence_probs,
            traits,
            genotype_data,
            phenotype_data,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sim_founders(
        ctx: &GpuContext,
        n_entries: usize,
        n_chromosomes: usize,
        n_loci: usize,
        n_traits: usize,
        ploidy: usize,
        with_sex: bool,
        seed: u64,
    ) -> Result<(Self, MapperG2P)> {
        let mut founders: Self = Self::new(ctx, n_entries)?;
        founders.entries = sim_entries(n_entries)?;
        founders.ploidy = ploidy;
        founders.genome = sim_genome(n_chromosomes, with_sex)?;
        founders.traits = sim_traits(n_traits)?;
        founders.sexes = sim_sexes(n_entries, with_sex, seed + 1)?;
        founders.loci = sim_loci(&founders.genome, n_loci, seed + 2)?;
        founders.haplotype_persistence_probs =
            haplotype_persistence_probabilities(&founders.genome, &founders.loci)?;
        founders.genotype_data = sim_genotype_data(
            ctx,
            &founders.entries,
            &founders.genome,
            &founders.loci,
            &founders.sexes,
            founders.ploidy,
            seed + 3,
        )?;
        let (phenotype_data, mapper) = sim_phenotype_data(
            ctx,
            &founders.genome,
            &founders.loci,
            &founders.genotype_data,
            &founders.traits,
            ploidy,
            seed + 4,
        )?;
        founders.phenotype_data = phenotype_data;
        Ok((founders, mapper))
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
        ensure!(
            mapper.heritabilities.len() == mapper.traits.len(),
            "The number of heritabilities does not match the number of traits in the genotype-to-phenotype mapper!"
        );
        // Locus data
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
        // Mating Pair data
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
        // Extract output
        let genotype_data = GpuTensor::from_buffer(
            std::sync::Arc::new(offspring_genotypes_buf),
            &[n_offsprings as u32, n_loci_alleles as u32, 2],
            None,
            None,
        )?;
        let phenotype_data = calc_phenotypes(ctx, &genotype_data, mapper, self.ploidy, seed)?;
        // Output
        let mut offsprings: Self = Self::new(ctx, n_offsprings)?;
        for (i, &(p1, p2)) in mating_pairs.iter().enumerate() {
            offsprings.entries[i] = self.entries[p1].clone();
            offsprings.entries[i].name =
                format!("{}--x--{}", self.entries[p1].name, self.entries[p2].name);
        }
        offsprings.ploidy = self.ploidy;
        let idx_sex_chromosome: Option<usize> =
            self.genome.iter().position(|x| x.is_sex_chromosome);
        if let Some(idx_sex_chromosome) = idx_sex_chromosome {
            let mut idx_range_sex_chrom = (0, 0);
            for locus in self.loci.iter() {
                if locus.chromosome_id == idx_sex_chromosome {
                    // Assumes the first locus in the sex chromosome defines the sex which is currently internally
                    // consistent with the expectation of only 2 alleles on the sex chromosome pair, i.e. for XX/ZZ and XY/ZW sexes
                    ensure!(
                        locus.col_idx.len() == 2,
                        "Sex chromosome locus must be biallelic!"
                    );
                    idx_range_sex_chrom = (locus.col_idx[0], locus.col_idx[0] + 2);
                    break;
                }
            }
            ensure!(
                idx_range_sex_chrom.1 > idx_range_sex_chrom.0,
                "Failed to locate a sex chromosome locus!"
            );
            let g_at_sex_chrom = genotype_data
                .slice_view(&[(0, n_offsprings), idx_range_sex_chrom, (0, 2)])?
                .to_vec_f32(ctx)?;
            offsprings.sexes = Vec::with_capacity(n_offsprings);
            for i in 0..n_offsprings {
                // let x0 = g_at_sex_chrom[(i * 4) + (2 * 0) + 0];
                // let x1 = g_at_sex_chrom[(i * 4) + (2 * 0) + 1];
                // let y0 = g_at_sex_chrom[(i * 4) + (2 * 1) + 0];
                // let y1 = g_at_sex_chrom[(i * 4) + (2 * 1) + 1];
                let y0 = g_at_sex_chrom[(i * 4) + 2];
                let y1 = g_at_sex_chrom[(i * 4) + 3];
                if (y0 + y1) > 0.0 {
                    offsprings.sexes.push(Sex::Heterogametic);
                } else {
                    offsprings.sexes.push(Sex::Homogametic);
                }
                // Sex::Homogametic => {
                //     // Homogametic individuals (e.g. XX or ZZ):
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_1)] += (ploidy / 2) as f32;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_1) + 1] +=
                //         (ploidy / 2) as f32;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_2)] += 0.0;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_2) + 1] += 0.0;
                // }
                // Sex::Heterogametic => {
                //     // Heterogametic individuals (e.g. XY or ZW).
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_1)] += (ploidy / 2) as f32;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_1) + 1] += 0.0;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_2)] += 0.0;
                //     genotype_data_tmp[idx_entry + (2 * idx_allele_2) + 1] +=
                //         (ploidy / 2) as f32;
                // }
            }
        } else {
            offsprings.sexes = vec![Sex::Hermaphrodite; n_offsprings];
        }
        offsprings.genome = self.genome.clone();
        offsprings.loci = self.loci.clone();
        offsprings.haplotype_persistence_probs = self.haplotype_persistence_probs.clone();
        offsprings.traits = self.traits.clone();
        offsprings.genotype_data = genotype_data;
        offsprings.phenotype_data = phenotype_data;
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
    fn founder_dimensions_are_consistent() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 10, 2, true, 42).unwrap();

        assert!(dataset.check_dimensions().is_ok());
    }

    #[test]
    fn sim_founders_is_deterministic() {
        let ctx = context();

        let (a, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let (b, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        assert_eq!(a.sexes, b.sexes);
        assert_eq!(a.genome, b.genome);
        assert_eq!(a.loci, b.loci);
        assert_eq!(a.traits, b.traits);

        assert_eq!(
            a.genotype_data.to_vec_f32(&ctx).unwrap(),
            b.genotype_data.to_vec_f32(&ctx).unwrap(),
        );

        assert_eq!(
            a.phenotype_data.to_vec_f32(&ctx).unwrap(),
            b.phenotype_data.to_vec_f32(&ctx).unwrap(),
        );
    }

    #[test]
    fn genotype_tensor_shape_matches_loci() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 10, 2, true, 42).unwrap();

        let n_loci_alleles: usize = dataset.loci.iter().map(|x| x.col_idx.len()).sum();

        assert_eq!(
            dataset.genotype_data.shape,
            vec![100, n_loci_alleles as u32, 2,]
        );
    }

    #[test]
    fn phenotype_tensor_shape_matches_traits() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 250, 5, 100, 7, 2, true, 42).unwrap();

        assert_eq!(dataset.phenotype_data.shape, vec![250, 7],);
    }

    #[test]
    fn sex_vector_matches_entry_count() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 500, 5, 100, 3, 2, true, 42).unwrap();

        assert_eq!(dataset.entries.len(), dataset.sexes.len(),);
    }

    #[test]
    fn sample_mating_pairs_is_deterministic() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let a = dataset.sample_mating_pairs(1000, 999).unwrap();

        let b = dataset.sample_mating_pairs(1000, 999).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn sample_mating_pairs_returns_requested_number() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = dataset.sample_mating_pairs(1234, 999).unwrap();

        assert_eq!(pairs.len(), 1234);
    }

    #[test]
    fn sampled_parents_follow_sex_constraints() {
        let ctx = context();

        let (dataset, _) = Dataset::sim_founders(&ctx, 1000, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = dataset.sample_mating_pairs(5000, 999).unwrap();

        for (p1, p2) in pairs {
            assert!(matches!(
                dataset.sexes[p1],
                Sex::Homogametic | Sex::Hermaphrodite
            ));

            assert!(matches!(
                dataset.sexes[p2],
                Sex::Heterogametic | Sex::Hermaphrodite
            ));
        }
    }

    #[test]
    fn mating_is_deterministic() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(50, 123).unwrap();

        let a = parents.mate(&ctx, pairs.clone(), &mapper, 999).unwrap();

        let b = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        assert_eq!(
            a.genotype_data.to_vec_f32(&ctx).unwrap(),
            b.genotype_data.to_vec_f32(&ctx).unwrap()
        );
    }

    #[test]
    fn offspring_dimensions_are_consistent() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(100, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        assert!(offspring.check_dimensions().is_ok());
    }

    #[test]
    fn offspring_retain_genome_structure() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(100, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        assert_eq!(offspring.genome, parents.genome,);

        assert_eq!(offspring.loci, parents.loci,);

        assert_eq!(offspring.traits, parents.traits,);

        assert_eq!(offspring.ploidy, parents.ploidy,);
    }

    #[test]
    fn offspring_names_record_parentage() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 100, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(100, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs.clone(), &mapper, 999).unwrap();

        for (i, (p1, p2)) in pairs.iter().enumerate() {
            let expected = format!(
                "{}--x--{}",
                parents.entries[*p1].name, parents.entries[*p2].name,
            );

            assert_eq!(offspring.entries[i].name, expected,);
        }
    }

    #[test]
    fn offspring_sexes_contain_only_valid_categories() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 1000, 5, 200, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(100, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        assert!(
            offspring
                .sexes
                .iter()
                .all(|x| { matches!(x, Sex::Homogametic | Sex::Heterogametic) })
        );
    }

    #[test]
    fn offspring_sex_ratio_is_approximately_half() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 2000, 5, 100, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(10000, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        let n_hetero = offspring
            .sexes
            .iter()
            .filter(|&&x| x == Sex::Heterogametic)
            .count();

        let p = n_hetero as f32 / offspring.sexes.len() as f32;

        assert!((p - 0.5).abs() < 0.05);
    }

    #[test]
    fn inferred_sex_matches_sex_locus() {
        let ctx = context();

        let (parents, mapper) = Dataset::sim_founders(&ctx, 1000, 5, 200, 5, 2, true, 42).unwrap();

        let pairs = parents.sample_mating_pairs(1000, 123).unwrap();

        let offspring = parents.mate(&ctx, pairs, &mapper, 999).unwrap();

        let idx_sex_chromosome = offspring
            .genome
            .iter()
            .position(|x| x.is_sex_chromosome)
            .unwrap();

        let sex_locus = offspring
            .loci
            .iter()
            .find(|l| l.chromosome_id == idx_sex_chromosome)
            .unwrap();

        let g = offspring
            .genotype_data
            .slice_view(&[
                (0, offspring.entries.len()),
                (sex_locus.col_idx[0], sex_locus.col_idx[0] + 2),
                (0, 2),
            ])
            .unwrap()
            .to_vec_f32(&ctx)
            .unwrap();

        for i in 0..offspring.entries.len() {
            let x0 = g[i * 4];
            let x1 = g[(i * 4) + 1];
            let y0 = g[(i * 4) + 2];
            let y1 = g[(i * 4) + 3];

            match offspring.sexes[i] {
                Sex::Homogametic => {
                    assert!(x0 > 0.0 || x1 > 0.0);
                    assert_eq!(y0 + y1, 0.0);
                }
                Sex::Heterogametic => {
                    assert!(x0 > 0.0 || x1 > 0.0);
                    assert!(y0 + y1 > 0.0);
                }
                Sex::Hermaphrodite => {
                    panic!("Unexpected hermaphroditic offspring")
                }
            }
        }
    }
}
