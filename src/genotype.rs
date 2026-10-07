use crate::entry::*;
use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, bail, ensure};
use rand::RngExt;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp};

#[derive(Debug, Clone, PartialEq)]
pub struct Chromosome {
    pub name: String,
    pub length: usize, // chromosome length (i.e. length of the largest homologous chromosome per pair)
    pub ld_decay_distance: usize, // will be used in mating assuming r(d) = exp(-d/L), where d is the distance between a pair of loci in bases and L is ld_decay_distance.
    pub is_sex_chromosome: bool,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Locus {
    pub chromosome_id: usize, // index of the chromosome containing this locus, which assumes one or more chromosomes are stored in a vector (contiguous/ordered list)
    pub position: usize,      // position in the chromosome
    pub alleles: Vec<String>, // sequence of each allele
    pub length: usize, // maximum size of alleles, i.e. the number of bases of the longest allele
    pub col_idx: Vec<usize>, // The column indices in the main genotype tensor, each referring to an allele
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sex {
    Hermaphrodite,
    Homogametic,
    Heterogametic,
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
                // NOTE: currently will have the same allele frequencies across the entire sex chromosome for each non-hermaphrodite sex, i.e. homogametic and heterogametic entries
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn sim_genome_rejects_zero_chromosomes() {
        assert!(sim_genome(0, false).is_err());
    }

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    #[test]
    fn sim_genome_returns_requested_number() {
        let genome = sim_genome(17, false).unwrap();

        assert_eq!(genome.len(), 17);
    }

    #[test]
    fn sim_genome_has_no_sex_chromosome_when_disabled() {
        let genome = sim_genome(10, false).unwrap();

        assert!(genome.iter().all(|x| !x.is_sex_chromosome));
    }

    #[test]
    fn sim_genome_has_single_sex_chromosome_when_enabled() {
        let genome = sim_genome(10, true).unwrap();

        let sex_chromosomes = genome.iter().filter(|x| x.is_sex_chromosome).count();

        assert_eq!(sex_chromosomes, 1);

        assert!(genome.last().unwrap().is_sex_chromosome);
    }

    #[test]
    fn sim_genome_names_are_unique() {
        let genome = sim_genome(100, false).unwrap();

        let names: HashSet<_> = genome.iter().map(|x| &x.name).collect();

        assert_eq!(names.len(), genome.len());
    }

    #[test]
    fn sim_sexes_is_deterministic() {
        let a = sim_sexes(100, true, 42).unwrap();
        let b = sim_sexes(100, true, 42).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn sim_sexes_returns_only_hermaphrodites_without_sex_system() {
        let sexes = sim_sexes(100, false, 42).unwrap();

        assert!(sexes.iter().all(|x| *x == Sex::Hermaphrodite));
    }

    #[test]
    fn sim_sexes_are_approximately_balanced() {
        let sexes = sim_sexes(10_000, true, 42).unwrap();

        let n_hetero = sexes.iter().filter(|&&x| x == Sex::Heterogametic).count();

        let p = n_hetero as f32 / sexes.len() as f32;

        assert!((p - 0.5).abs() < 0.05);
    }

    #[test]
    fn sim_loci_rejects_zero() {
        let genome = sim_genome(5, false).unwrap();

        assert!(sim_loci(&genome, 0, 42).is_err());
    }

    #[test]
    fn sim_loci_requires_at_least_one_locus_per_chromosome() {
        let genome = sim_genome(5, false).unwrap();

        assert!(sim_loci(&genome, 4, 42).is_err());
    }

    #[test]
    fn sim_loci_returns_requested_number() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        assert_eq!(loci.len(), 100);
    }

    #[test]
    fn sim_loci_is_deterministic() {
        let genome = sim_genome(5, false).unwrap();

        let a = sim_loci(&genome, 100, 42).unwrap();
        let b = sim_loci(&genome, 100, 42).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn every_chromosome_receives_at_least_one_locus() {
        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        for chromosome_id in 0..genome.len() {
            assert!(loci.iter().any(|l| l.chromosome_id == chromosome_id));
        }
    }

    #[test]
    fn locus_column_indices_are_contiguous() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        let mut indices: Vec<usize> = loci
            .iter()
            .flat_map(|l| l.col_idx.iter().copied())
            .collect();

        indices.sort_unstable();

        for (i, idx) in indices.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }

    #[test]
    fn locus_column_indices_are_unique() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        let indices: Vec<usize> = loci
            .iter()
            .flat_map(|l| l.col_idx.iter().copied())
            .collect();

        let unique: HashSet<_> = indices.iter().copied().collect();

        assert_eq!(indices.len(), unique.len());
    }

    #[test]
    fn sex_chromosome_loci_are_biallelic() {
        let genome = sim_genome(5, true).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        let idx_sex = genome.iter().position(|x| x.is_sex_chromosome).unwrap();

        for locus in loci.iter() {
            if locus.chromosome_id == idx_sex {
                assert_eq!(locus.alleles.len(), 2);
            }
        }
    }

    #[test]
    fn autosomal_loci_have_between_two_and_five_alleles() {
        let genome = sim_genome(5, true).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        let idx_sex = genome.iter().position(|x| x.is_sex_chromosome).unwrap();

        for locus in loci.iter() {
            if locus.chromosome_id != idx_sex {
                assert!((2..=5).contains(&locus.alleles.len()));
            }
        }
    }

    #[test]
    fn haplotype_persistence_probabilities_match_locus_count() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();

        assert_eq!(probs.len(), loci.len());
    }

    #[test]
    fn haplotype_persistence_probabilities_are_bounded() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();

        assert!(probs.iter().all(|&x| (0.5..=1.0).contains(&x)));
    }

    #[test]
    fn first_haplotype_persistence_probability_is_half() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();

        assert_eq!(probs[0], 0.5);
    }

    #[test]
    fn chromosome_boundaries_have_probability_half() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 100, 42).unwrap();

        let probs = haplotype_persistence_probabilities(&genome, &loci).unwrap();

        for i in 1..loci.len() {
            if loci[i - 1].chromosome_id != loci[i].chromosome_id {
                assert_eq!(probs[i], 0.5);
            }
        }
    }

    #[test]
    fn loci_positions_are_non_decreasing_within_chromosomes() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

        for chromosome_id in 0..genome.len() {
            let positions: Vec<usize> = loci
                .iter()
                .filter(|l| l.chromosome_id == chromosome_id)
                .map(|l| l.position)
                .collect();

            for i in 1..positions.len() {
                assert!(positions[i] >= positions[i - 1]);
            }
        }
    }

    #[test]
    fn loci_positions_are_within_chromosome_bounds() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

        for locus in loci.iter() {
            let chromosome = &genome[locus.chromosome_id];

            assert!(locus.position < chromosome.length);
        }
    }

    #[test]
    fn locus_allele_names_match_allele_count() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        for locus in loci.iter() {
            assert_eq!(locus.alleles.len(), locus.col_idx.len());
        }
    }

    #[test]
    fn loci_have_positive_lengths() {
        let genome = sim_genome(5, false).unwrap();

        let loci = sim_loci(&genome, 200, 42).unwrap();

        assert!(loci.iter().all(|l| l.length > 0));
    }

    #[test]
    fn chromosome_names_are_unique() {
        let genome = sim_genome(100, false).unwrap();

        let names: HashSet<_> = genome.iter().map(|c| &c.name).collect();

        assert_eq!(names.len(), genome.len());
    }

    #[test]
    fn sex_chromosome_is_last_chromosome() {
        let genome = sim_genome(10, true).unwrap();

        let idx = genome.iter().position(|x| x.is_sex_chromosome).unwrap();

        assert_eq!(idx, genome.len() - 1);
    }

    fn allele_frequencies(genotype: &[f32], n_entries: usize, n_loci_alleles: usize) -> Vec<f32> {
        let mut freqs = Vec::with_capacity(n_loci_alleles);

        for allele in 0..n_loci_alleles {
            let mut dosage = 0.0;

            for entry in 0..n_entries {
                let base = entry * n_loci_alleles * 2;

                dosage += genotype[base + (2 * allele)];

                dosage += genotype[base + (2 * allele) + 1];
            }

            freqs.push(dosage / ((2 * n_entries) as f32));
        }

        freqs
    }

    #[test]
    fn founder_allele_frequencies_are_u_shaped() {
        let ctx = context();

        let entries = sim_entries(1_000).unwrap();

        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

        let sexes = sim_sexes(entries.len(), false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let g = genotype_data.to_vec_f32(&ctx).unwrap();

        let freqs = allele_frequencies(&g, entries.len(), genotype_data.shape[1] as usize);

        let low = freqs.iter().filter(|&&p| p < 0.2).count();

        let high = freqs.iter().filter(|&&p| p > 0.8).count();

        let mid = freqs.iter().filter(|&&p| (0.4..0.6).contains(&p)).count();

        assert!(low + high > mid);
    }

    #[test]
    fn founder_allele_frequencies_are_edge_enriched() {
        let ctx = context();

        let entries = sim_entries(1_000).unwrap();

        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

        let sexes = sim_sexes(entries.len(), false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let g = genotype_data.to_vec_f32(&ctx).unwrap();

        let freqs = allele_frequencies(&g, entries.len(), genotype_data.shape[1] as usize);

        let edge = freqs.iter().filter(|&&p| p < 0.25 || p > 0.75).count();

        let centre = freqs.iter().filter(|&&p| (0.4..0.6).contains(&p)).count();

        assert!(edge > centre);
    }

    #[test]
    fn first_allele_is_on_average_most_common() {
        let ctx = context();

        let entries = sim_entries(1_000).unwrap();

        let genome = sim_genome(10, false).unwrap();

        let loci = sim_loci(&genome, 500, 42).unwrap();

        let sexes = sim_sexes(entries.len(), false, 42).unwrap();

        let genotype_data =
            sim_genotype_data(&ctx, &entries, &genome, &loci, &sexes, 2, 123).unwrap();

        let g = genotype_data.to_vec_f32(&ctx).unwrap();

        let freqs = allele_frequencies(&g, entries.len(), genotype_data.shape[1] as usize);

        let mut first = Vec::new();
        let mut others = Vec::new();

        for locus in &loci {
            if locus.col_idx.len() < 2 {
                continue;
            }

            first.push(freqs[locus.col_idx[0]]);

            for idx in &locus.col_idx[1..] {
                others.push(freqs[*idx]);
            }
        }

        let mean_first = first.iter().sum::<f32>() / first.len() as f32;

        let mean_other = others.iter().sum::<f32>() / others.len() as f32;

        assert!(
            mean_first > mean_other,
            "mean_first={} mean_other={}",
            mean_first,
            mean_other
        );
    }
}
