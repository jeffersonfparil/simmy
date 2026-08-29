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
