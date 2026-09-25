use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use rand::RngExt;
use rand::prelude::IndexedRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp, Normal};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub name: String,
    pub lengths: (usize, usize), // homologous chromosome lengths
    pub centromere_positions: (usize, usize), // centromere positions in corresponding homologous chromosomes
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
    pub is_sex: bool,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub species: String,
    pub ploidy: usize,
    pub group: String,
    pub notes: String,
}

#[derive(Debug)]
pub struct Data {
    pub entries: Vec<Entry>,
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
        writeln!(f, "\t- Chromosomes: {}", self.genome.len())?;
        writeln!(f, "\t- Loci: {}", self.loci.len())?;
        writeln!(f, "\t- Locus Alleles: {}", n_loci_alleles)?;
        writeln!(f, "\t- Traits: {}", self.traits.len())?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Genotype Tensor Shape: {}", self.genotype_data)?;
        writeln!(f, "\t  ---------------------------------")?;
        writeln!(f, "\t- Phenotype Tensor Shape: {}", self.phenotype_data)?;
        writeln!(f, "\t  ---------------------------------")?;
        let sex_chromosomes = self.genome.iter().filter(|c| c.is_sex_chromosome).count();
        let sex_traits = self.traits.iter().filter(|t| t.is_sex).count();
        writeln!(f, "\t- Sex Chromosomes: {}", sex_chromosomes)?;
        writeln!(f, "\t- Sex Traits: {}", sex_traits)?;
        if let Some(entry) = self.entries.first() {
            writeln!(f, "\t- Ploidy: {}", entry.ploidy)?;
        }
        Ok(())
    }
}

impl Data {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &GpuContext,
        n_entries: usize,
        n_chromosomes: usize,
        n_loci: usize,
        n_traits: usize,
        with_sex: bool,
        ploidy: usize,
        seed: usize,
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
                ploidy,
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
                centromere_positions: (500_000, 500_000),
                ld_decay_distance: 10_000,
                is_sex_chromosome: with_sex && (i == (n_chromosomes - 1)),
            });
        }
        // Loci
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        // Divy up the loci into chromosomes
        let mut positions_per_chromosome: Vec<Vec<usize>> = Vec::with_capacity(n_chromosomes);
        let mut m = n_loci / n_chromosomes;
        for (i, chromosome) in genome.iter().enumerate() {
            m += if i < (n_chromosomes - 1) {
                0
            } else {
                // Move remaining loci to the last chromosome for simplicity
                n_loci - (m * n_chromosomes)
            };
            let n = chromosome.lengths.0;
            let mut pos: Vec<usize> = Vec::with_capacity(m);
            for j in 0..m {
                pos.push(j * n / m);
            }
            positions_per_chromosome.push(pos);
        }
        // Simulate loci-alleles
        let mut loci: Vec<Locus> = Vec::with_capacity(n_loci);
        let mut n_loci_alleles: usize = 0;
        let exponential = Exp::new(1.0).expect("Failed to initialise exponential distribution!");
        for (i, positions) in positions_per_chromosome.iter().enumerate() {
            for &pos in positions {
                let n_alleles: usize = if !genome[i].is_sex_chromosome {
                    // TODO: add more alleles but for now we only have SNPs, i.e. A, T, C, G, and D (note that these are still SNPs even though we can have multi-allelic loci because the variations remain single nucleotides)
                    let a: f64 = exponential.sample(&mut rng);
                    a.round().clamp(2.0, 5.0) as usize
                } else {
                    2
                };
                loci.push(Locus {
                    chromosome_id: i,
                    position: pos,
                    alleles: ["A", "T", "C", "G", "D"][0..n_alleles]
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
                is_sex: with_sex && (i == (n_traits - 1)),
                description: "".to_owned(),
            });
        }
        // Genotype data
        let mut genotype_data_tmp: Vec<f32> = vec![0.0; n_entries * n_loci_alleles * 2];
        let beta_n =
            Beta::new(2.0, 5.0).expect("Failed to initialise a Beta distribution (a=2; b=5)!"); // We will sample allele indices per locus from a skewed distribution, i.e. biased towards the first allele which is more realistic than uniform distribution.
        let beta_u =
            Beta::new(0.5, 0.5).expect("Failed to initialise a Beta distribution (a=b=0.5)!"); // We will sample from a realistic U shaped distribution for genomewide loci on a single population.
        for i in 0..n_entries {
            let idx = i * 2 * n_loci_alleles;
            let is_monogametic = with_sex && beta_u.sample(&mut rng) < 0.5; // XX or ZZ, i.e. define whether the entry is monogametic or not
            for locus in &loci {
                let n_alleles = locus.col_idx.len();
                if !genome[locus.chromosome_id].is_sex_chromosome {
                    // Sample alleles and their dosages per homologous chromosome or parent,
                    // where the same allele may be sampled and hence fixed on a homologous chromosome and so we add to the initialised 0.0 dosages.
                    for j in 0..2 {
                        // For autosomal loci
                        // Each allele from each ploidy is sampled here
                        for _ in 0..(ploidy / 2) {
                            let allele_1_parent_j = locus.col_idx[(beta_n.sample(&mut rng)
                                * ((n_alleles - 1) as f32))
                                .round()
                                as usize];
                            let allele_2_parent_j = locus.col_idx[(beta_n.sample(&mut rng)
                                * ((n_alleles - 1) as f32))
                                .round()
                                as usize];
                            let allele_1_dosage_parent_j = (beta_u.sample(&mut rng) as f32).round();
                            let allele_2_dosage_parent_j = 1.00 - allele_1_dosage_parent_j;
                            genotype_data_tmp[idx + (2 * allele_1_parent_j) + j] +=
                                allele_1_dosage_parent_j;
                            genotype_data_tmp[idx + (2 * allele_2_parent_j) + j] +=
                                allele_2_dosage_parent_j;
                        }
                    }
                } else {
                    // For sex chromosome loci, i.e. with 2 alleles which represent the heterogametic thing-o
                    // XX or ZZ homogametic and XY or ZW heterogametic ==> males and female sexes
                    let allele_1_parent_j = locus.col_idx[0];
                    let allele_2_parent_j = locus.col_idx[1];
                    if is_monogametic {
                        // XX or ZZ
                        genotype_data_tmp[idx + (2 * allele_1_parent_j)] += (ploidy / 2) as f32;
                        genotype_data_tmp[idx + (2 * allele_1_parent_j) + 1] += 0.0;
                        genotype_data_tmp[idx + (2 * allele_2_parent_j)] += (ploidy / 2) as f32;
                        genotype_data_tmp[idx + (2 * allele_2_parent_j) + 1] += 0.0;
                    } else {
                        // XY or ZW
                        genotype_data_tmp[idx + (2 * allele_1_parent_j)] += (ploidy / 2) as f32;
                        genotype_data_tmp[idx + (2 * allele_1_parent_j) + 1] += 0.0;
                        genotype_data_tmp[idx + (2 * allele_2_parent_j)] += 0.0;
                        genotype_data_tmp[idx + (2 * allele_2_parent_j) + 1] += (ploidy / 2) as f32;
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
        let mut phenotype_data_tmp: Vec<f32> = vec![0.0; n_entries * n_traits];
        let normal =
            Normal::new(0.0, 1.0).expect("Failed to initialise a standard normal distribution!");
        for i in 0..n_entries {
            let is_monogametic = if !with_sex {
                false
            } else {
                let sex_locus = loci
                    .iter()
                    .rev()
                    .find(|&l| genome[l.chromosome_id].is_sex_chromosome)
                    .expect("No sex locus found despite with_sex=true");
                let idx_0 = genotype_data.linear_index(&[i, sex_locus.col_idx[0], 0]);
                let idx_1 = genotype_data.linear_index(&[i, sex_locus.col_idx[1], 0]);
                genotype_data_tmp[idx_0] == genotype_data_tmp[idx_1]
            };
            for j in 0..n_traits {
                if with_sex && (j == (n_traits - 1)) {
                    // Sex, i.e. 0 for monogametic (XX or ZZ) and 1 for heterogametic (XY or ZW)
                    phenotype_data_tmp[(i * n_traits) + j] = if is_monogametic { 0.0 } else { 1.0 };
                } else {
                    // Continuous trait values
                    phenotype_data_tmp[(i * n_traits) + j] = normal.sample(&mut rng);
                }
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
            genome,
            loci,
            traits,
            genotype_data,
            phenotype_data,
        })
    }
    pub fn check_dimensions(&self, ctx: &GpuContext) -> Result<()> {
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
        let genotype = self.genotype_data.to_vec_f32(ctx)?;
        let n_loci_alleles: usize = self.loci.iter().map(|l| l.col_idx.len()).sum();
        let mut ploidy: usize = 0;
        for entry_idx in 0..self.entries.len() {
            let base = entry_idx * n_loci_alleles * 2;
            for locus in &self.loci {
                let total_dosage: usize = locus
                    .col_idx
                    .iter()
                    .map(|&col| {
                        (genotype[base + (2 * col)] + genotype[base + (2 * col) + 1]) as usize
                    })
                    .sum();
                ensure!(
                    total_dosage.is_multiple_of(2),
                    "We do not support odd ploidies at the moment!"
                );
                ploidy = if ploidy == 0 { total_dosage } else { ploidy };
                ensure!(
                    ploidy == total_dosage,
                    "Ploidies are inconsistent across entries and/or loci!"
                );
            }
        }
        Ok(())
    }
    pub fn sample_mating_pairs(
        &self,
        ctx: &GpuContext,
        n_offsprings: usize,
        seed: u64,
    ) -> Result<Vec<(usize, usize)>> {
        self.check_dimensions(ctx)?;
        let n_entries: usize = self.entries.len();
        let n_traits: usize = self.traits.len();
        let with_sex: bool = self
            .genome
            .last()
            .is_some_and(|last_chrom| last_chrom.is_sex_chromosome);
        ensure!(
            !with_sex || self.traits.last().is_some_and(|t| t.is_sex),
            "Sex chromosome detected but final trait is not marked as a sex trait!"
        );
        let mut mating_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_offsprings);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (idx_homogametics, idx_heterogametics) = if with_sex {
            let phenotypes: Vec<f32> = self.phenotype_data.to_vec_f32(ctx)?;
            let mut idx_homogametics = vec![];
            let mut idx_heterogametics = vec![];
            for i in 0..n_entries {
                let sex = phenotypes[(i * n_traits) + (n_traits - 1)];
                match sex {
                    0.0 => idx_homogametics.push(i),
                    1.0 => idx_heterogametics.push(i),
                    _ => anyhow::bail!(
                        "Invalid sex phenotype value: {}. We only expect 0.0 for homogametic and 1.0 for heterogametic!",
                        sex
                    ),
                }
            }
            (idx_homogametics, idx_heterogametics)
        } else {
            (vec![], vec![])
        };
        for _ in 0..n_offsprings {
            if !with_sex {
                // Monoecious
                mating_pairs.push((
                    rng.random_range(0..n_entries),
                    rng.random_range(0..n_entries),
                ));
            } else {
                // Dioecious
                ensure!(
                    !idx_homogametics.is_empty(),
                    "No homogametic individuals available!"
                );
                ensure!(
                    !idx_heterogametics.is_empty(),
                    "No heterogametic individuals available!"
                );
                if let (Some(&x), Some(&y)) = (
                    idx_homogametics.choose(&mut rng),
                    idx_heterogametics.choose(&mut rng),
                ) {
                    mating_pairs.push((x, y));
                }
            }
        }
        Ok(mating_pairs)
    }
    // TODO: mating with LD
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
        assert!(Data::new(&ctx, 0, 1, 1, 1, false, 2, 42).is_err());
    }
    #[test]
    fn rejects_zero_chromosomes() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 0, 1, 1, false, 2, 42).is_err());
    }
    #[test]
    fn rejects_zero_loci() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 0, 1, false, 2, 42).is_err());
    }
    #[test]
    fn rejects_fewer_loci_than_chromosomes() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 10, 5, 1, false, 2, 42).is_err());
    }
    #[test]
    fn rejects_zero_traits() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 0, false, 2, 42).is_err());
    }
    #[test]
    fn rejects_zero_ploidy() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 1, false, 0, 42).is_err());
    }
    #[test]
    fn rejects_odd_ploidy() {
        let ctx = context();
        assert!(Data::new(&ctx, 1, 1, 1, 1, false, 3, 42).is_err());
    }
    #[test]
    fn creates_expected_counts() {
        let ctx = context();
        let data = Data::new(&ctx, 25, 5, 100, 7, false, 2, 42).unwrap();
        println!("data: {}", data);
        assert_eq!(data.entries.len(), 25);
        assert_eq!(data.genome.len(), 5);
        assert_eq!(data.loci.len(), 100);
        assert_eq!(data.traits.len(), 7);
    }
    #[test]
    fn creates_one_locus_per_chromosome_minimum() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 8, 8, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        let mut counts = [0usize; 8];
        for locus in &data.loci {
            counts[locus.chromosome_id] += 1;
        }
        assert!(counts.iter().all(|&n| n > 0));
    }
    #[test]
    fn last_chromosome_is_sex_chromosome_when_enabled() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 50, 1, true, 2, 42).unwrap();
        println!("data: {}", data);
        assert!(!data.genome[0].is_sex_chromosome);
        assert!(!data.genome[1].is_sex_chromosome);
        assert!(!data.genome[2].is_sex_chromosome);
        assert!(!data.genome[3].is_sex_chromosome);
        assert!(data.genome[4].is_sex_chromosome);
    }
    #[test]
    fn no_sex_chromosomes_when_disabled() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 50, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        assert!(data.genome.iter().all(|c| !c.is_sex_chromosome));
    }
    #[test]
    fn last_trait_is_sex_trait_when_enabled() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 1, 10, 5, true, 2, 42).unwrap();
        println!("data: {}", data);
        assert!(data.traits[4].is_sex);
        assert!(data.traits[..4].iter().all(|t| !t.is_sex));
    }
    #[test]
    fn locus_col_indices_are_contiguous_and_unique() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        let mut all = data
            .loci
            .iter()
            .flat_map(|l| l.col_idx.iter().copied())
            .collect::<Vec<_>>();
        all.sort_unstable();
        for (i, idx) in all.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }
    #[test]
    fn locus_allele_count_is_between_two_and_five() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        for locus in &data.loci {
            assert!((2..=5).contains(&locus.alleles.len()));
            assert_eq!(locus.length, 1);
        }
    }
    #[test]
    fn locus_column_count_matches_allele_count() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        for locus in &data.loci {
            assert_eq!(locus.alleles.len(), locus.col_idx.len());
        }
    }
    #[test]
    fn chromosome_positions_are_sorted() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
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
    fn reproducible_with_same_seed() {
        let ctx = context();
        let data1 = Data::new(&ctx, 10, 5, 100, 3, true, 2, 123).unwrap();
        let data2 = Data::new(&ctx, 10, 5, 100, 3, true, 2, 123).unwrap();
        assert_eq!(data1.entries.len(), data2.entries.len());
        assert_eq!(data1.genome.len(), data2.genome.len());
        assert_eq!(data1.loci.len(), data2.loci.len());
        for (a, b) in data1.loci.iter().zip(data2.loci.iter()) {
            assert_eq!(a.chromosome_id, b.chromosome_id);
            assert_eq!(a.position, b.position);
            assert_eq!(a.alleles, b.alleles);
            assert_eq!(a.col_idx, b.col_idx);
        }
    }
    #[test]
    fn entry_names_are_unique() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 1, false, 2, 42).unwrap();
        println!("data: {}", data);
        let mut names = data
            .entries
            .iter()
            .map(|e| e.name.clone())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 100);
    }
    #[test]
    fn trait_names_are_unique() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 20, false, 2, 42).unwrap();
        println!("data: {}", data);
        let mut names = data
            .traits
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 20);
    }
    #[test]
    fn locus_total_allele_dosage_equals_ploidy() {
        let ctx = context();
        for ploidy in [2usize, 4, 6, 8, 10] {
            let data = Data::new(&ctx, 10, 5, 100, 1, false, ploidy, 42).unwrap();
            println!("data: {}", data);
            let genotype = data.genotype_data.to_vec_f32(&ctx).unwrap();
            let n_loci_alleles: usize = data.loci.iter().map(|l| l.col_idx.len()).sum();
            for entry_idx in 0..data.entries.len() {
                let base = entry_idx * n_loci_alleles * 2;
                for locus in &data.loci {
                    let total: usize = locus
                        .col_idx
                        .iter()
                        .map(|&col| {
                            (genotype[base + (2 * col)] + genotype[base + (2 * col) + 1]) as usize
                        })
                        .sum();
                    assert_eq!(
                        total, ploidy,
                        "entry={}, chromosome={}, position={}, ploidy={}",
                        entry_idx, locus.chromosome_id, locus.position, ploidy
                    );
                }
            }
        }
    }
    #[test]
    fn homologous_chromosome_dosage_equals_half_ploidy() {
        let ctx = context();
        for ploidy in [2usize, 4, 6, 8, 10] {
            let data = Data::new(&ctx, 10, 5, 100, 1, false, ploidy, 42).unwrap();
            println!("data: {}", data);
            let genotype = data.genotype_data.to_vec_f32(&ctx).unwrap();
            let n_loci_alleles: usize = data.loci.iter().map(|l| l.col_idx.len()).sum();
            for entry_idx in 0..data.entries.len() {
                let base = entry_idx * n_loci_alleles * 2;
                for locus in &data.loci {
                    for parent in 0..2 {
                        let dosage: usize = locus
                            .col_idx
                            .iter()
                            .map(|&col| genotype[base + (2 * col) + parent] as usize)
                            .sum();
                        assert_eq!(
                            dosage,
                            ploidy / 2,
                            "entry={}, chromosome={}, position={}, parent={}, ploidy={}",
                            entry_idx,
                            locus.chromosome_id,
                            locus.position,
                            parent,
                            ploidy
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn check_dimensions_passes_for_valid_data() {
        let ctx = context();
        let data = Data::new(
            &ctx, 10,  // entries
            5,   // chromosomes
            100, // loci
            3,   // traits
            true, 4, // ploidy
            42,
        )
        .unwrap();
        println!("data: {}", data);
        assert!(data.check_dimensions(&ctx).is_ok());
    }
    #[test]
    fn sex_chromosome_genotypes_are_xx_or_xy() {
        let ctx = context();
        let ploidy = 4usize;
        let data = Data::new(&ctx, 50, 5, 100, 1, true, ploidy, 42).unwrap();
        println!("data: {}", data);
        let genotype = data.genotype_data.to_vec_f32(&ctx).unwrap();
        let sex_chr = data
            .genome
            .iter()
            .position(|c| c.is_sex_chromosome)
            .unwrap();
        let sex_loci: Vec<_> = data
            .loci
            .iter()
            .filter(|l| l.chromosome_id == sex_chr)
            .collect();
        let n_loci_alleles: usize = data.loci.iter().map(|l| l.col_idx.len()).sum();
        for entry_idx in 0..data.entries.len() {
            let base = entry_idx * n_loci_alleles * 2;
            for locus in &sex_loci {
                let a0 = locus.col_idx[0];
                let a1 = locus.col_idx[1];
                let x0 = genotype[base + (2 * a0)] as usize;
                let x1 = genotype[base + (2 * a0) + 1] as usize;
                let y0 = genotype[base + (2 * a1)] as usize;
                let y1 = genotype[base + (2 * a1) + 1] as usize;
                let is_xx = x0 == ploidy / 2 && x1 == 0 && y0 == ploidy / 2 && y1 == 0;
                let is_xy = x0 == ploidy / 2 && x1 == 0 && y0 == 0 && y1 == ploidy / 2;
                assert!(
                    is_xx || is_xy,
                    "invalid sex genotype: ({},{},{},{})",
                    x0,
                    x1,
                    y0,
                    y1
                );
            }
        }
    }
    #[test]
    fn sex_chromosome_loci_have_two_alleles() {
        let ctx = context();
        let data = Data::new(&ctx, 10, 5, 100, 1, true, 2, 42).unwrap();
        println!("data: {}", data);
        let sex_chr = data
            .genome
            .iter()
            .position(|c| c.is_sex_chromosome)
            .unwrap();
        for locus in data.loci.iter().filter(|l| l.chromosome_id == sex_chr) {
            assert_eq!(locus.alleles.len(), 2);
            assert_eq!(locus.col_idx.len(), 2);
        }
    }
    #[test]
    fn sex_trait_is_binary() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 3, true, 2, 42).unwrap();
        println!("data: {}", data);
        let phenotype = data.phenotype_data.to_vec_f32(&ctx).unwrap();
        let sex_trait_idx = data.traits.len() - 1;
        for entry_idx in 0..data.entries.len() {
            let sex = phenotype[(entry_idx * data.traits.len()) + sex_trait_idx];
            assert!(sex == 0.0 || sex == 1.0);
        }
    }
    #[test]
    fn sex_trait_matches_sex_genotype() {
        let ctx = context();
        let ploidy = 4usize;
        let data = Data::new(&ctx, 100, 5, 100, 3, true, ploidy, 42).unwrap();
        println!("data: {}", data);
        let genotype = data.genotype_data.to_vec_f32(&ctx).unwrap();
        let phenotype = data.phenotype_data.to_vec_f32(&ctx).unwrap();
        let sex_trait_idx = data.traits.len() - 1;
        let sex_chr = data
            .genome
            .iter()
            .position(|c| c.is_sex_chromosome)
            .unwrap();
        let sex_locus = data
            .loci
            .iter()
            .find(|l| l.chromosome_id == sex_chr)
            .unwrap();
        let n_loci_alleles: usize = data.loci.iter().map(|l| l.col_idx.len()).sum();
        for entry_idx in 0..data.entries.len() {
            let base = entry_idx * n_loci_alleles * 2;
            let a0 = sex_locus.col_idx[0];
            let a1 = sex_locus.col_idx[1];
            let x0 = genotype[base + (2 * a0)] as usize;
            let y0 = genotype[base + (2 * a1)] as usize;
            let expected = if x0 == ploidy / 2 && y0 == ploidy / 2 {
                0.0
            } else {
                1.0
            };
            let observed = phenotype[(entry_idx * data.traits.len()) + sex_trait_idx];
            assert_eq!(
                observed, expected,
                "entry={}, x0={}, y0={}",
                entry_idx, x0, y0
            );
        }
    }
    #[test]
    fn sex_ratio_is_approximately_fifty_fifty() {
        let ctx = context();
        let data = Data::new(&ctx, 1000, 5, 100, 3, true, 2, 42).unwrap();
        println!("data: {}", data);
        let phenotype = data.phenotype_data.to_vec_f32(&ctx).unwrap();
        let sex_trait_idx = data.traits.len() - 1;
        let mut monogametic = 0usize;
        let mut heterogametic = 0usize;
        for entry_idx in 0..data.entries.len() {
            let sex = phenotype[(entry_idx * data.traits.len()) + sex_trait_idx];
            if sex == 0.0 {
                monogametic += 1;
            } else if sex == 1.0 {
                heterogametic += 1;
            } else {
                panic!("invalid sex phenotype: {}", sex);
            }
        }
        let total = monogametic + heterogametic;
        let mono_fraction = monogametic as f64 / total as f64;
        let hetero_fraction = heterogametic as f64 / total as f64;
        assert!(
            (mono_fraction - 0.5).abs() < 0.10,
            "monogametic fraction {} not close to 0.5",
            mono_fraction
        );
        assert!(
            (hetero_fraction - 0.5).abs() < 0.10,
            "heterogametic fraction {} not close to 0.5",
            hetero_fraction
        );
    }
    #[test]
    fn check_dimensions_rejects_trait_tensor_mismatch() {
        let ctx = context();
        let mut data = Data::new(&ctx, 10, 5, 100, 3, false, 2, 42).unwrap();
        data.traits.pop();
        let err = data.check_dimensions(&ctx).unwrap_err();
        assert!(
            err.to_string()
                .contains("traits in `traits` and `phenotype_data` do not match")
        );
    }
    #[test]
    fn mating_pair_count_matches_request() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 3, false, 2, 42).unwrap();
        let n_offsprings = 1_000;
        let pairs = data.sample_mating_pairs(&ctx, n_offsprings, 123).unwrap();
        assert_eq!(pairs.len(), n_offsprings);
    }
    #[test]
    fn mating_pairs_contain_valid_entry_indices() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 3, false, 2, 42).unwrap();
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(&ctx, 10_000, 123).unwrap();
        for (a, b) in pairs {
            assert!(a < data.entries.len());
            assert!(b < data.entries.len());
        }
    }
    #[test]
    fn mating_pairs_are_reproducible() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 3, true, 2, 42).unwrap();
        println!("data: {}", data);
        let pairs_1 = data.sample_mating_pairs(&ctx, 1_000, 999).unwrap();
        let pairs_2 = data.sample_mating_pairs(&ctx, 1_000, 999).unwrap();
        assert_eq!(pairs_1, pairs_2);
    }
    #[test]
    fn dioecious_pairings_are_homogametic_by_heterogametic() {
        let ctx = context();
        let data = Data::new(&ctx, 500, 5, 100, 3, true, 2, 42).unwrap();
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(&ctx, 10_000, 999).unwrap();
        let phenotype = data.phenotype_data.to_vec_f32(&ctx).unwrap();
        let sex_col = data.traits.len() - 1;
        for (x, y) in pairs {
            let sex_x = phenotype[(x * data.traits.len()) + sex_col];
            let sex_y = phenotype[(y * data.traits.len()) + sex_col];
            assert_eq!(sex_x, 0.0);
            assert_eq!(sex_y, 1.0);
        }
    }
    #[test]
    fn monoecious_population_can_self() {
        let ctx = context();
        let data = Data::new(&ctx, 50, 5, 100, 3, false, 2, 42).unwrap();
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(&ctx, 20_000, 123).unwrap();
        assert!(pairs.iter().any(|(a, b)| a == b));
    }
    #[test]
    fn zero_offspring_returns_empty_vector() {
        let ctx = context();
        let data = Data::new(&ctx, 100, 5, 100, 3, true, 2, 42).unwrap();
        println!("data: {}", data);
        let pairs = data.sample_mating_pairs(&ctx, 0, 123).unwrap();
        assert!(pairs.is_empty());
    }
    #[test]
    fn invalid_sex_phenotype_is_rejected() {
        let ctx = context();
        let mut data = Data::new(&ctx, 20, 5, 100, 3, true, 2, 42).unwrap();
        let mut phenotype = data.phenotype_data.to_vec_f32(&ctx).unwrap();
        let entry_idx = 0;
        let sex_col = data.traits.len() - 1;
        phenotype[(entry_idx * data.traits.len()) + sex_col] = 2.0;
        data.phenotype_data = GpuTensor::from_f32(
            &ctx,
            &phenotype,
            &[data.entries.len() as u32, data.traits.len() as u32],
            None,
            None,
        )
        .unwrap();
        assert!(data.sample_mating_pairs(&ctx, 10, 123).is_err());
    }
}
