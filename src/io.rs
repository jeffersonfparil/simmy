use crate::linalg::context::GpuContext;
use crate::linalg::kernel::GpuKernel;
use crate::linalg::tensor::GpuTensor;

pub struct Chromosomes {
    chromosomes: Vec<String>, // names of each chromosome or scaffold or contig
    lengths: Vec<usize>, // size in bp of each chromosome or scaffold or contig
}

pub struct Alleles {
    names: Vec<String>, // names of the alleles, e.g. SNP: "A", "T", "C", "G", "DEL", and largest variants: "GATGCGC", "ACTAGCTAGCTA", "GCGCGAG"
}

pub struct Locus {
    chromosome_id: usize,
    start: usize,
    end: usize,
    allele_ids: Vec<usize>,
}

pub struct LocusAllele {
    locus_id: usize,
    allele_id: usize,
}

// For checking the validity of the genomic information read/loaded/simulated/generated
pub struct Genome {
    chromosomes: Chromosomes,
    alleles: Alleles,
    loci: Vec<Locus>,
    loci_alleles: Vec<LocusAllele>,
}

pub struct Entries {
    names: Vec<String>,
    species: Vec<String>,
    population: Vec<String>,
    classification: Vec<String>,
    notes: Vec<String>,
}

pub struct Traits {
    names: Vec<String>,
    notes: Vec<String>,
}


// Main genotype data struct
pub struct GenotypeData {
    entry_ids: Vec<usize>,
    locus_allele_ids: Vec<usize>,
    data: GpuTensor,
}

// Main phenotype data struct
pub struct PhenotypeData {
    entry_ids: Vec<usize>,
    trait_ids: Vec<usize>,
    data: GpuTensor,
}