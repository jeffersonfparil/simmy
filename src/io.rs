//! # Genomic and Phenotypic Data Structures for Breeding Simulations
//!
//! This module defines the CPU-side relational metadata and structural foundations
//! of the simulation engine. The primary design goal is to cleanly decouple heavy,
//! text-based biological metadata (which resides on the CPU) from raw, highly
//! optimized numerical arrays (which reside on the GPU as [`GpuTensor`] instances) [1, 2].
//!
//! ### Why This Architecture is Optimized for Breeding Simulations:
//! 1. **GPU/CPU Modularity:** Simulating cohorts over generations involves complex CPU-bound
//!    recombination, crossover logic, and pedigree tracing. Meanwhile, calculating genomic
//!    breeding values (GEBVs), selection indices, and linkage disequilibrium (LD) matrices
//!    is delegated to massive parallel matrix algebra on the GPU [2].
//! 2. **Support for Multi-Allelic Loci:** Real-world breeding pools contain highly variable
//!    multi-allelic states (e.g., microsatellites, structural variants, or multiple founder
//!    haplotypes). By flattening these states into a relational mapping table ([`LocusAllele`]),
//!    this design supports arbitrary allelic counts per site on a unified GPU matrix coordinate system.
//! 3. **Struct-of-Arrays (SoA) Layout:** Storing metadata attributes in parallel vectors
//!    allows rapid CPU-side scanning, demographic filtering, and generation masking without
//!    the memory overhead of unpacking deeply nested structures.
//!

use crate::linalg::tensor::GpuTensor;
use anyhow::{Result, ensure};

/// Represents physical genomic chromosomes, scaffolds, or contigs.
///
/// ### Breeding Simulation Context:
/// Essential for simulating physical linkage, chromosomal crossover events during
/// meiosis, and modeling genetic recombination maps. The sequence coordinates mapped
/// against physical chromosomal lengths enable calculation of centimorgan (cM) distances
/// and crossover probabilities during simulated mating cycles.
#[derive(Debug, Clone)]
pub struct Chromosomes {
    /// Unique names/identifiers for each chromosome, scaffold, or contig.
    pub chromosomes: Vec<String>,
    /// Physical sizes (lengths in base pairs) corresponding to each chromosome.
    /// Used for validating recombination crossover boundaries.
    pub lengths: Vec<usize>,
}

impl Chromosomes {
    pub fn new(n: usize, lengths: Option<Vec<usize>>) -> Result<Self> {
        let lengths = match lengths {
            Some(x) => x,
            None => vec![1_000_000; n],
        };
        ensure!(
            n == lengths.len(),
            "The number of names (n={}) and lengths (n={}) must match!",
            n,
            lengths.len()
        );
        let chromosomes: Vec<String> = (0..n).map(|i| format!("chr_{}", i)).collect();
        Ok(Self {
            chromosomes,
            lengths,
        })
    }
}

/// A global dictionary of unique allelic variant sequences or sequence states.
///
/// ### Breeding Simulation Context:
/// This acts as a centralized registry for any physical allele represented in the pool—ranging
/// from single-nucleotide polymorphisms (SNPs) to complex insertions, deletions (DEL),
/// and large structural variants. It decouples descriptive string-based sequence data
/// from the active, high-speed numeric matrices running on the GPU.
#[derive(Debug, Clone)]
pub struct Alleles {
    /// String representations of the allele sequences (e.g., "A", "T", "DEL", "GATGCGC").
    pub names: Vec<String>,
}

const SNPS: &[&str] = &["A", "T", "C", "G", "DEL"];

impl Alleles {
    pub fn new(n: usize, names: Option<Vec<String>>) -> Result<Self> {
        let names = match names {
            Some(x) => x,
            None => {
                let mut names: Vec<String> = Vec::with_capacity(n);
                // For n <= 5: names  in vec!["A", "T", "C", "G", "DEL"]
                // For n <= 10: names in vec!["A", "T", "C", "G", "DEL", "TA", "TT", "TC", "TG", "TDEL"]
                // For n <= 15: names in vec!["A", "T", "C", "G", "DEL", "TA", "TT", "TC", "TG", "TDEL", "CA", "CT", "CC", "CG", "CDEL"]
                // For n <= 20: names in vec!["A", "T", "C", "G", "DEL", "TA", "TT", "TC", "TG", "TDEL", "CA", "CT", "CC", "CG", "CDEL", "GA", "GT", "GC", "GG", "GDEL"]
                // i.e. little-endian generation because this is simpler than the big-endian
                for i in 0..n {
                    let mut name_components: Vec<&str> = Vec::new();
                    let mut idx = i;
                    loop {
                        let snp_idx = idx % SNPS.len();
                        idx /= SNPS.len(); // floor of corresponding float quotients
                        name_components.push(SNPS[snp_idx]);
                        if idx == 0 {
                            break;
                        }
                    }
                    name_components.reverse();
                    names.push(name_components.join(""));
                }
                names
            }
        };
        ensure!(
            n == names.len(),
            "The numbe of names (n={}) and names (n={}) must match!",
            n,
            names.len()
        );
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by_key(|&i| names[i].to_owned());
        for i in 1..n {
            let idx_0 = perm[i - 1];
            let idx_1 = perm[i];
            ensure!(
                names[idx_0] != names[idx_1],
                "Duplicated allele: {}!",
                names[idx_0]
            );
        }
        Ok(Self { names })
    }
}

/// Defines a physical genomic feature or coordinate region (locus) and its valid alleles.
///
/// ### Breeding Simulation Context:
/// Represents individual markers (SNPs) or quantitative trait loci (QTL). By holding
/// a list of valid `allele_ids` local to this locus, it naturally supports **monoallelic** (fixed),
/// **biallelic** (standard SNPs), and **multi-allelic** loci within the same genome.
#[derive(Debug, Clone)]
pub struct Locus {
    /// ID of the chromosome where this locus resides (indexes into [`Chromosomes`]).
    pub chromosome_id: usize,
    /// Start physical coordinate in base pairs (0-indexed, inclusive).
    pub start: usize,
    /// End physical coordinate in base pairs (exclusive).
    pub end: usize,
    /// Valid allele identifiers observable at this locus (indexes into [`Alleles`]).
    pub allele_ids: Vec<usize>,
}

/// A key relational mapping that bridges a physical locus to a specific sequence variant.
///
/// ### Why We Chose This Structure:
/// In population simulations, alleles are variable per locus. A standard matrix representation
/// assuming only biallelic SNPs fails under multi-allelic states.
///
/// This structure solves the problem by providing a flat relational lookup table.
/// Every entry represents a unique **locus-allele combination**, which directly maps to a column
/// index in the GPU genotype tensor. This enables the GPU compute kernels to perform rapid
/// linear algebra on variable-allele genomes by representing them as flattened dosage columns.
#[derive(Debug, Clone)]
pub struct LocusAllele {
    /// The physical location of the marker (indexes into [`Genome::loci`]).
    pub locus_id: usize,
    /// The specific sequence variant associated with this locus (indexes into [`Alleles`]).
    pub allele_id: usize,
}

/// The global blueprint of the simulation's genomic architecture.
///
/// ### Breeding Simulation Context:
/// Acts as the central validation authority on the CPU. It ensures that any imported,
/// generated, or simulated genomic structure is internally consistent, verifying that
/// chromosomal boundaries, loci, and locus-allele combinations are valid before initiating
/// a simulation run.
#[derive(Debug, Clone)]
pub struct Genome {
    /// The chromosome configurations defining physical linkage groups.
    pub chromosomes: Chromosomes,
    /// The global lookup dictionary of physical sequence variations.
    pub alleles: Alleles,
    /// The list of genomic loci/markers (such as QTL or SNPs) being tracked.
    pub loci: Vec<Locus>,
    /// The relational map of all locus-allele combinations (used to decode GPU tensor column indices).
    pub loci_alleles: Vec<LocusAllele>,
}

/// High-level demographic metadata representing individual animals, plants, or lines.
///
/// ### Breeding Simulation Context:
/// Manages population structures, cohort generations, and pedigrees on the CPU. By separating
/// this qualitative tracking from the dense numerical genotype tensors, you can query, slice,
/// and filter breeding cohorts (e.g., separating founders, generation F1, or target breeding lines)
/// cleanly on the CPU to dynamically assemble indexing vectors for GPU acceleration.
#[derive(Debug, Clone)]
pub struct Entries {
    /// Names or unique identifiers of each individual or line in the dataset.
    pub names: Vec<String>,
    /// Taxonomic classification (e.g., species or subspecies) for multi-species scenarios.
    pub species: Vec<String>,
    /// Breeding cohort or origin group identifier (e.g., "Founder_A", "Cycle_5").
    pub population: Vec<String>,
    /// User-defined categorization, breeding tiers, or selection groups.
    pub classification: Vec<String>,
    /// Arbitrary historical logs, pedigree descriptions, or metadata notes.
    pub notes: Vec<String>,
}

/// Metadata describing quantitative traits under selection.
///
/// ### Breeding Simulation Context:
/// Defines targets for breeding programs (e.g., disease resistance, yield, stature).
/// It enables multi-trait selection schemes, economic weights configuration, and tracking
/// how genetic architectures map to multiple target phenotypes.
#[derive(Debug, Clone)]
pub struct Traits {
    /// Names of the tracked quantitative traits.
    pub names: Vec<String>,
    /// Descriptions of genetic parameters, heritabilities, or breeding objectives.
    pub notes: Vec<String>,
}

/// The primary GPU-backed genotype representation for high-throughput computing.
///
/// ### Why We Chose This Structure:
/// In quantitative genetics and breeding, the genotype matrix is the bottleneck of calculations.
/// Storing this as a [`GpuTensor`] on the GPU enables extremely fast, massively parallel operations:
/// - Calculating genomic relationship matrices (GRM).
/// - Matrix multiplication of marker effect sizes for genomic prediction ($X \beta$).
/// - Slicing and selecting specific subgroups using zero-copy stride manipulations [2, 3].
#[derive(Debug)]
pub struct GenotypeData {
    /// Rows of the genotype matrix: indices pointing to the evaluated individuals in [`Entries`].
    pub entry_ids: Vec<usize>,
    /// Columns of the genotype matrix: indices mapping to physical alleles via [`LocusAllele`].
    pub locus_allele_ids: Vec<usize>,
    /// Dense GPU matrix of shape `[entry_ids.len(), locus_allele_ids.len()]` [2].
    /// Represents allelic dosages (e.g., count, probability, or state of the allele).
    pub data: GpuTensor,
}

// impl GenotypeData {
//     pub fn new() {
//         todo!()
//     }
// }

/// The observed phenotype metrics backed by high-performance GPU storage.
///
/// ### Breeding Simulation Context:
/// This holds the quantitative performance values of each individual across multiple traits.
/// Storing these on the GPU allows the simulation engine to perform real-time selection-index
/// calculations, variance-covariance estimations, and evaluation sweeps directly in GPU memory,
/// feeding selection decisions straight back into the next simulated mating cycle.
#[derive(Debug)]
pub struct PhenotypeData {
    /// Rows of the phenotype matrix: indices pointing to the evaluated individuals in [`Entries`].
    pub entry_ids: Vec<usize>,
    /// Columns of the phenotype matrix: indices pointing to quantitative traits in [`Traits`].
    pub trait_ids: Vec<usize>,
    /// Dense GPU matrix of shape `[entry_ids.len(), trait_ids.len()]` [2].
    /// Stores the phenotypic value floats (e.g., breeding estimates, observed values).
    pub data: GpuTensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    // -----------------------------
    // Chromosomes::new tests
    // -----------------------------

    #[test]
    fn chromosomes_default_lengths() -> Result<()> {
        let chr = Chromosomes::new(3, None)?;
        assert_eq!(chr.chromosomes, vec!["chr_0", "chr_1", "chr_2"]);
        assert_eq!(chr.lengths, vec![1_000_000, 1_000_000, 1_000_000]);
        Ok(())
    }

    #[test]
    fn chromosomes_custom_lengths() -> Result<()> {
        let chr = Chromosomes::new(3, Some(vec![10, 20, 30]))?;
        assert_eq!(chr.chromosomes, vec!["chr_0", "chr_1", "chr_2"]);
        assert_eq!(chr.lengths, vec![10, 20, 30]);
        Ok(())
    }

    #[test]
    fn chromosomes_length_mismatch_fails() {
        let result = Chromosomes::new(3, Some(vec![10, 20]));
        assert!(result.is_err());
    }

    #[test]
    fn chromosomes_zero_n() -> Result<()> {
        let chr = Chromosomes::new(0, None)?;
        assert!(chr.chromosomes.is_empty());
        assert!(chr.lengths.is_empty());
        Ok(())
    }

    // -----------------------------
    // Alleles::new tests
    // -----------------------------

    #[test]
    fn alleles_default_n_leq_5() -> Result<()> {
        let a = Alleles::new(5, None)?;
        assert_eq!(a.names, vec!["A", "T", "C", "G", "DEL"]);
        Ok(())
    }

    #[test]
    fn alleles_default_n_10() -> Result<()> {
        let a = Alleles::new(10, None)?;
        assert_eq!(
            a.names,
            vec!["A", "T", "C", "G", "DEL", "TA", "TT", "TC", "TG", "TDEL"]
        );
        Ok(())
    }

    #[test]
    fn alleles_default_n_20() -> Result<()> {
        let a = Alleles::new(20, None)?;
        // Check first 5 and last 5 only
        assert_eq!(&a.names[..5], &["A", "T", "C", "G", "DEL"]);
        assert_eq!(&a.names[15..20], &["GA", "GT", "GC", "GG", "GDEL"]);
        Ok(())
    }

    #[test]
    fn alleles_custom_names() -> Result<()> {
        let custom = vec!["X".into(), "Y".into(), "Z".into()];
        let a = Alleles::new(3, Some(custom.clone()))?;
        assert_eq!(a.names, custom);
        Ok(())
    }

    #[test]
    fn alleles_custom_name_length_mismatch_fails() {
        let result = Alleles::new(3, Some(vec!["A".into(), "B".into()]));
        assert!(result.is_err());
    }

    #[test]
    fn alleles_no_duplicates() -> Result<()> {
        let a = Alleles::new(50, None)?;
        let mut sorted = a.names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), a.names.len());
        Ok(())
    }

    #[test]
    fn alleles_large_n() -> Result<()> {
        let a = Alleles::new(500, None)?;
        assert_eq!(a.names.len(), 500);
        // Check that the last allele is multi-character
        assert!(a.names[499].len() >= 3);
        Ok(())
    }
}
