use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution, Uniform};

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
    pub chromosome_id: u64, // index of the chromosome containing this locus, which assumes one or more chromosomes are stored in a vector (contiguous/ordered list)
    pub position: u64, // position in the chromosome
    pub alleles: Vec<String>, // sequence of each allele
    pub length: u64, // maximum size of alleles, i.e. the number of bases of the longest allele
    pub col_idx: Vec<u64>, // The column indices in the main genotype tensor, each referring to an allele
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub species: String,
    pub ploidy: u64,
    pub group: String,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct GenotypeData {
    pub genome: Vec<Chromosome>,
    pub loci: Vec<Locus>,
    pub entries: Vec<Entry>,
}