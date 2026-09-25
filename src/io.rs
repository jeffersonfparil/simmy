use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::Result;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Distribution, Exp};

// use anyhow::{Context, Result, ensure};
// use rand::prelude::*;
// use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
// use rand_distr::{Beta, Distribution, Uniform, Exp};

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub name: String,
    pub lengths: (u64, u64),
    pub centromere_positions: (u64, u64),
    pub ld_decay_distance: u64,
    pub is_sex_chromosome: bool,
}

#[derive(Debug, Clone)]
pub struct Locus {
    pub chromosome_id: usize, // index of the chromosome containing this locus, which assumes one or more chromosomes are stored in a vector (contiguous/ordered list)
    pub position: u64,        // position in the chromosome
    pub alleles: Vec<String>, // sequence of each allele
    pub length: u64, // maximum size of alleles, i.e. the number of bases of the longest allele
    pub col_idx: Vec<u64>, // The column indices in the main genotype tensor, each referring to an allele
}

#[derive(Debug, Clone)]
pub struct Trait {
    _name: String,
    _is_sex: bool,
    _description: String,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub species: String,
    pub ploidy: u64,
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
    pub fn new(
        _ctx: GpuContext,
        n_entries: usize,
        n_chromosomes: usize,
        n_loci: usize,
        _n_traits: usize,
        _with_sex: bool,
        seed: u64,
    ) -> Result<Self> {
        // Entries
        let mut entries: Vec<Entry> = Vec::with_capacity(n_entries);
        let n_digits: usize = format!("{}", n_entries).len();
        for i in 0..n_entries {
            entries.push(Entry {
                name: format!("entry_{:0>n_digits$}", i),
                species: "".to_owned(),
                ploidy: 2,
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
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Divy up the loci into chromosomes
        let mut positions_per_chromosome: Vec<Vec<u64>> = Vec::with_capacity(n_chromosomes);
        let mut m = (n_loci / n_chromosomes) as u64;
        for (i, chromosome) in genome.iter().enumerate() {
            m += if i < (n_chromosomes - 1) {
                0
            } else {
                (n_loci as u64) - (m * n_chromosomes as u64)
            };
            let n = chromosome.lengths.0;
            let k = n / m;
            let mut pos: Vec<u64> = (0..n).step_by(k as usize).collect();
            if (pos.len() as u64) < m {
                pos.push(n - 1);
            }
            positions_per_chromosome.push(pos);
        }
        let exponential = Exp::new(1.0).expect("Failed to initialise exponential distribution!");
        let mut loci: Vec<Locus> = Vec::with_capacity(n_loci);
        let mut locus_allele_counter: u64 = 0;
        for (i, positions) in positions_per_chromosome.iter().enumerate() {
            for &pos in positions {
                let n_alleles = {
                    let a: f64 = exponential.sample(&mut rng);
                    a.round().max(2.0) as usize
                };
                loci.push(Locus {
                    chromosome_id: i,
                    position: pos,
                    alleles: ["A", "T", "C", "G", "D"][0..n_alleles]
                        .iter()
                        .map(|&x| x.to_owned())
                        .collect::<Vec<String>>(),
                    length: 1,
                    col_idx: (locus_allele_counter..(locus_allele_counter + (n_alleles as u64)))
                        .collect::<Vec<u64>>(),
                });
                locus_allele_counter += n_alleles as u64;
            }
        }
        todo!()
    }
}
