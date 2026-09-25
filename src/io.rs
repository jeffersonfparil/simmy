use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Exp, Normal};

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub name: String,
    pub lengths: (usize, usize),
    pub centromere_positions: (usize, usize),
    pub ld_decay_distance: usize,
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
    pub genotype_data: GpuTensor,
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
                is_sex_chromosome: i < (n_chromosomes - 1),
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
                n_loci - (m * n_chromosomes)
            };
            let n = chromosome.lengths.0;
            let k = n / m;
            let mut pos: Vec<usize> = (0..n).step_by(k).collect();
            if pos.len() < m {
                pos.push(n - 1);
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
                }; // TODO: add more alleles but for now we only have SNPs, i.e. A, T, C, G, and D
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
        let beta =
            Beta::new(0.5, 0.5).expect("Failed to initialise a Beta distribution (a=b=0.5)!");
        for _i in 0..n_entries {
            for _j in 0..n_loci_alleles {
                let n_alleles_from_parent_1 = (beta.sample(&mut rng) * (ploidy as f32)).round();
                let n_alleles_from_parent_2 = (beta.sample(&mut rng) * (ploidy as f32)).round();
                genotype_data_tmp.push(n_alleles_from_parent_1);
                genotype_data_tmp.push(n_alleles_from_parent_2);
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
        for _i in 0..n_entries {
            for _j in 0..n_traits {
                phenotype_data_tmp.push(normal.sample(&mut rng));
            }
        }
        let phenotype_data = GpuTensor::from_f32(
            ctx,
            &phenotype_data_tmp,
            &[n_entries as u32, n_loci_alleles as u32, 2],
            None,
            None,
        )?;
        Ok(Self {
            entries,
            genome,
            loci,
            traits,
            genotype_data,
            phenotype_data,
        })
    }
}
