use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp, Normal};

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
    pub genotype_data: GpuTensor, // shape: n_entries x n_loci_alleles x maternal+paternal haplotypes
    pub phenotype_data: GpuTensor,
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
                let n_alleles = {
                    let a: f64 = exponential.sample(&mut rng);
                    a.round().clamp(2.0, 5.0) as usize
                }; // TODO: add more alleles but for now we only have SNPs, i.e. A, T, C, G, and D (note that these are still SNPs even though we can have multi-allelic loci because the variations remain single nucleotides)
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
            for locus in &loci {
                let idx = i * 2 * n_loci_alleles;
                let n_alleles = locus.col_idx.len();
                for j in 0..2 {
                    let allele_1_parent_j = locus.col_idx
                        [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                    let allele_2_parent_j = locus.col_idx
                        [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                    let allele_1_dosage_parent_j =
                        (beta_u.sample(&mut rng) * ((ploidy / 2) as f32)).round();
                    let allele_2_dosage_parent_j = ((ploidy / 2) as f32) - allele_1_dosage_parent_j;
                    genotype_data_tmp[idx + (2 * allele_1_parent_j) + j] +=
                        allele_1_dosage_parent_j;
                    genotype_data_tmp[idx + (2 * allele_2_parent_j) + j] +=
                        allele_2_dosage_parent_j;
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
            genome,
            loci,
            traits,
            genotype_data,
            phenotype_data,
        })
    }
    // TODO: pmating pair selection
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
        assert_eq!(data.entries.len(), 25);
        assert_eq!(data.genome.len(), 5);
        assert_eq!(data.loci.len(), 100);
        assert_eq!(data.traits.len(), 7);
    }
    #[test]
    fn creates_one_locus_per_chromosome_minimum() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 8, 8, 1, false, 2, 42).unwrap();
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
        assert!(data.genome.iter().all(|c| !c.is_sex_chromosome));
    }
    #[test]
    fn last_trait_is_sex_trait_when_enabled() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 1, 10, 5, true, 2, 42).unwrap();
        assert!(data.traits[4].is_sex);
        assert!(data.traits[..4].iter().all(|t| !t.is_sex));
    }
    #[test]
    fn locus_col_indices_are_contiguous_and_unique() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
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
        for locus in &data.loci {
            assert!((2..=5).contains(&locus.alleles.len()));
            assert_eq!(locus.length, 1);
        }
    }
    #[test]
    fn locus_column_count_matches_allele_count() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
        for locus in &data.loci {
            assert_eq!(locus.alleles.len(), locus.col_idx.len());
        }
    }
    #[test]
    fn chromosome_positions_are_sorted() {
        let ctx = context();
        let data = Data::new(&ctx, 1, 5, 100, 1, false, 2, 42).unwrap();
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
}
