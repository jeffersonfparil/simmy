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
            Beta::new(0.5, 0.5).expect("Failed to initialise a Beta distribution (a=b=0.5)!"); // We will sample from a realistic U shaped distribution (bell-shaped assumes balancing selection on all loci which rarely occur in genome-wide loci).
        for i in 0..n_entries {
            for locus in &loci {
                let allele_dosage_from_parent_1 =
                    (beta_u.sample(&mut rng) * ((ploidy / 2) as f32)).round();
                let allele_dosage_from_parent_2 =
                    ((ploidy / 2) as f32) - allele_dosage_from_parent_1;
                let idx = i * 2 * n_loci_alleles;
                // We select an allele from each parent, where we may have multiple alleles per locus, hence parent1 ∈ {0,1} & parent2 ∈ {1,0} at a single locus is valid for a diploid because allele count sum to 2.
                let n_alleles = locus.col_idx.len();
                let j_1 = locus.col_idx
                    [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                let j_2 = locus.col_idx
                    [(beta_n.sample(&mut rng) * ((n_alleles - 1) as f32)).round() as usize];
                genotype_data_tmp[idx + (2 * j_1)] = allele_dosage_from_parent_1;
                genotype_data_tmp[idx + (2 * j_2) + 1] = allele_dosage_from_parent_2;
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
