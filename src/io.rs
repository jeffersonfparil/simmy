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

use std::range::Range;

use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_distr::{Beta, Distribution};

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
    // Distances in bp (L) between pairs of bases where the linkage (r²) falls to around 0.368, where:
    /// r²(d) = exp(-d / L), and d is the distance between 2 loci in bp
    pub ld_decay_distances: Vec<usize>,
}

impl Chromosomes {
    /// Constructs a `Chromosomes` collection containing `n` chromosomes with
    /// user-provided or default lengths and LD decay distances.
    ///
    /// # Overview
    ///
    /// If `lengths` is provided, the slice is copied into an owned
    /// `Vec<usize>` and validated to ensure its length matches `n`.
    ///
    /// If `lengths` is `None`, all chromosomes are assigned a default
    /// physical length of `1_000_000` base pairs.
    ///
    /// If `ld_decay_distances` is provided, the values are copied into an
    /// owned `Vec<usize>` and validated to ensure:
    ///
    /// * The number of decay distances equals `n`.
    /// * Every decay distance is greater than zero.
    ///
    /// If `ld_decay_distances` is `None`, all chromosomes are assigned a
    /// default LD decay distance of `2_000` bp.
    ///
    /// Chromosome names are generated deterministically using zero-padded
    /// numeric formatting. The padding width is derived from `n - 1`,
    /// ensuring consistent identifier width across all chromosomes. For
    /// example:
    ///
    /// ```text
    /// n = 3
    ///     chr_0
    ///     chr_1
    ///     chr_2
    ///
    /// n = 12
    ///     chr_00
    ///     chr_01
    ///     ...
    ///     chr_11
    /// ```
    ///
    /// This padding guarantees lexicographically ordered chromosome
    /// identifiers, simplifying metadata management and downstream tensor
    /// indexing.
    ///
    /// # LD Decay Model
    ///
    /// Each chromosome is assigned a characteristic LD decay distance `L`
    /// (in base pairs). Simmy assumes:
    ///
    /// ```text
    /// r²(d) = r²(0) exp(-d / L)
    /// ```
    ///
    /// where:
    ///
    /// * `r²(d)` is the expected linkage disequilibrium between two loci
    ///   separated by distance `d`.
    /// * `r²(0)` is assumed to be `1.0`.
    /// * `d` is the physical distance between loci in base pairs.
    /// * `L` is the chromosome-specific LD decay distance.
    ///
    /// Consequently, when:
    ///
    /// ```text
    /// d = L
    /// ```
    ///
    /// the expected LD falls to:
    ///
    /// ```text
    /// r²(L) = exp(-1) ≈ 0.368
    /// ```
    ///
    /// Larger values of `L` imply longer-range haplotype structure and
    /// slower LD decay. Smaller values imply shorter haplotype blocks and
    /// more rapid LD decay.
    ///
    /// # Parameters
    ///
    /// * `n` - Number of chromosomes to construct. Must be non-zero.
    /// * `lengths` - Optional chromosome lengths in base pairs.
    /// * `ld_decay_distances` - Optional chromosome-specific LD decay
    ///   distances in base pairs.
    ///
    /// # Returns
    ///
    /// A validated `Chromosomes` instance containing:
    ///
    /// * `chromosomes` - zero-padded chromosome identifiers.
    /// * `lengths` - physical chromosome lengths.
    /// * `ld_decay_distances` - chromosome-specific LD decay distances.
    ///
    /// # Validation
    ///
    /// * Ensures `n > 0`.
    /// * Ensures `lengths.len() == n`.
    /// * Ensures `ld_decay_distances.len() == n`.
    /// * Ensures all LD decay distances are greater than zero.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * `n == 0`.
    /// * The number of supplied chromosome lengths does not equal `n`.
    /// * The number of supplied LD decay distances does not equal `n`.
    /// * Any LD decay distance is zero.
    pub fn new(
        n: usize,
        lengths: Option<&[usize]>,
        ld_decay_distances: Option<&[usize]>,
    ) -> Result<Self> {
        ensure!(n > 0, "Number of chromosomes need to non-zero!");
        let lengths = match lengths {
            Some(x) => x.to_vec(),
            None => vec![1_000_000; n],
        };
        ensure!(
            n == lengths.len(),
            "The number of chromosomes (n={}) and lengths (n={}) must match!",
            n,
            lengths.len()
        );
        let ld_decay_distances: Vec<usize> = match ld_decay_distances {
            Some(x) => x.to_vec(),
            None => {
                vec![2_000; n]
            }
        };
        ensure!(
            ld_decay_distances.iter().all(|&d| d > 0),
            "Decay distances must be greater than zero!"
        );
        ensure!(
            n == ld_decay_distances.len(),
            "The number of chromosomes (n={}) and ld_decay_distances (n={}) must match!",
            n,
            ld_decay_distances.len()
        );
        let n_digits: usize = format!("{}", n - 1).len();
        let chromosomes: Vec<String> = (0..n).map(|i| format!("chr_{:0>n_digits$}", i)).collect();
        Ok(Self {
            chromosomes,
            lengths,
            ld_decay_distances,
        })
    }
}

/// A global dictionary of unique allelic variant sequences or sequence states.
///
/// ### Breeding Simulation Context:
/// This acts as a centralized registry for any physical allele represented in the pool—ranging
/// from single-nucleotide polymorphisms (SNPs) to complex insertions, deletions (D),
/// and large structural variants. It decouples descriptive string-based sequence data
/// from the active, high-speed numeric matrices running on the GPU.
#[derive(Debug, Clone)]
pub struct Alleles {
    /// String representations of the allele sequences (e.g., "A", "T", "D", "GATGCGC").
    pub sequences: Vec<String>,
}

const SNPS: &[&str] = &["A", "T", "C", "G", "D"];

impl Alleles {
    /// Constructs an `Alleles` registry containing `n` unique allele sequence names.
    ///
    /// # Overview
    /// If `sequences` is provided, the constructor copies the supplied slice of `&str`
    /// into an owned `Vec<String>` and verifies that its length matches `n`.
    ///
    /// If `sequences` is `None`, allele sequences are generated automatically using a
    /// little‑endian base‑`SNPS.len()` encoding over the SNP alphabet `SNPS = ["A","T","C","G","D"]`.
    /// This produces deterministic, unique allele strings such as:
    ///
    /// - `n <= 5`:     A, T, C, G, D
    /// - `n <= 10`:    A, T, C, G, D, TA, TT, TC, TG, TD
    /// - `n <= 15`:    (previous) + CA, CT, CC, CG, CD
    /// - `n <= 20`:    (previous) + GA, GT, GC, GG, GD
    ///
    /// The generation rule is effectively a mixed‑radix counter over the SNP alphabet,
    /// where the least‑significant “digit” varies fastest. Components are reversed to
    /// yield a big‑endian human‑readable allele string.
    ///
    /// # Validation
    /// - Ensures `sequences.len() == n`.
    /// - Ensures all allele sequences are unique by lexicographically sorting indices and
    ///   checking adjacent entries for equality.
    ///
    /// # Parameters
    /// - `n`: Number of alleles to construct.
    /// - `sequences`: Optional slice of allele sequences. If `None`, sequences are generated.
    ///
    /// # Returns
    /// A fully validated `Alleles` instance containing `n` unique allele sequences.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `n == 0`,
    /// - The number of provided sequences does not equal `n`.
    /// - Any duplicate allele name is detected after sorting.
    pub fn new(n: usize, sequences: Option<&[&str]>) -> Result<Self> {
        ensure!(n > 0, "Number of alleles need to non-zero!");
        let sequences = match sequences {
            Some(x) => x.iter().map(|&xi| xi.to_owned()).collect::<Vec<String>>(),
            None => {
                let mut sequences: Vec<String> = Vec::with_capacity(n);
                // For n <= 5: sequences  in &["A", "T", "C", "G", "D"]
                // For n <= 10: sequences in &["A", "T", "C", "G", "D", "TA", "TT", "TC", "TG", "TD"]
                // For n <= 15: sequences in &["A", "T", "C", "G", "D", "TA", "TT", "TC", "TG", "TD", "CA", "CT", "CC", "CG", "CD"]
                // For n <= 20: sequences in &["A", "T", "C", "G", "D", "TA", "TT", "TC", "TG", "TD", "CA", "CT", "CC", "CG", "CD", "GA", "GT", "GC", "GG", "GD"]
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
                    sequences.push(name_components.join(""));
                }
                sequences
            }
        };
        ensure!(
            n == sequences.len(),
            "The number of requested sequences (n={}) and sequences (sequences.len()={}) must match!",
            n,
            sequences.len()
        );
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by_key(|&i| sequences[i].to_owned());
        for i in 1..n {
            let idx_0 = perm[i - 1];
            let idx_1 = perm[i];
            ensure!(
                sequences[idx_0] != sequences[idx_1],
                "Duplicated allele: {}!",
                sequences[idx_0]
            );
        }
        Ok(Self { sequences })
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

impl Locus {
    /// Generates `n` loci distributed proportionally across chromosomes and
    /// constructs both the locus list and the flattened locus‑allele mapping.
    ///
    /// # Overview
    /// Loci are allocated to chromosomes in proportion to their physical lengths.
    /// Within each chromosome, loci are placed at uniform intervals using:
    ///
    /// `step = chromosome_length / loci_on_chromosome`
    ///
    /// For each locus:
    /// - A random number of alleles in `1..=total_alleles` is selected.
    /// - Allele IDs are sampled uniformly from the global allele registry.
    /// - The locus width is set to the maximum sequence length among its alleles.
    ///
    /// Coordinates are clamped so that `(start, end)` always lies within the
    /// chromosome boundary. If `start + locus_length` would exceed the chromosome
    /// length, the locus is shifted left to end exactly at the boundary.
    ///
    /// # Output Structures
    /// Returns:
    /// - `Vec<Locus>`: physical loci with clamped coordinates and allele sets.
    /// - `Vec<LocusAllele>`: flattened `(locus_id, allele_id)` pairs suitable
    ///   for GPU genotype tensor construction.
    ///
    /// # Guarantees
    /// - Total number of loci equals `n`.
    /// - Every locus has at least one allele.
    /// - No locus exceeds chromosome boundaries.
    ///
    /// # Caveats
    /// - Chromosomes may receive zero loci if proportional allocation rounds to zero.
    /// - Loci may overlap if `locus_length > step`, since spacing is uniform but
    ///   allele sequence lengths vary.
    ///
    /// # Parameters
    /// - `chromosomes`: Chromosome names and lengths.
    /// - `alleles`: Global allele registry; allele names determine sequence length.
    /// - `n`: Total number of loci to generate.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `n == 0`,
    /// - total genome length is insufficient to place `n` loci of width `max_allele_length`.
    pub fn new(
        chromosomes: &Chromosomes,
        alleles: &Alleles,
        n: usize,
        seed: u64,
    ) -> Result<(Vec<Locus>, Vec<LocusAllele>)> {
        ensure!(n > 0, "Number of loci need to non-zero!");
        let total_length: usize = chromosomes.lengths.iter().sum();
        let max_allele_length: usize = alleles.sequences.iter().map(|x| x.len()).max().unwrap_or(1);
        ensure!(
            total_length >= n * max_allele_length,
            "Requested maximum length (n*max_allele_length={}bp) is greater than the total genome length (chromosomes.lengths.iter().sum()={})!",
            n * max_allele_length,
            total_length
        );
        let total_alleles: usize = alleles.sequences.len();
        let n_chromosomes: usize = chromosomes.chromosomes.len();
        let mut loci_per_chromosome: Vec<usize> = (0..n_chromosomes)
            .map(|i| {
                (((n * max_allele_length * chromosomes.lengths[i]) as f64) / (total_length as f64))
                    .floor() as usize
            })
            .collect();
        let m: usize = loci_per_chromosome.iter().sum();
        loci_per_chromosome[n_chromosomes - 1] = if m < n {
            loci_per_chromosome[n_chromosomes - 1] + (n - m)
        } else {
            loci_per_chromosome[n_chromosomes - 1]
        };
        // This may drop whole chromosomes if they are too small.
        // Some loci may overlap due to allele sequence lengths.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut out_loci: Vec<Self> = Vec::with_capacity(n);
        let mut out_loci_alleles: Vec<LocusAllele> = Vec::with_capacity(n);
        for (i, &loci) in loci_per_chromosome.iter().enumerate().take(n_chromosomes) {
            let length: usize = chromosomes.lengths[i];
            if loci == 0 {
                continue;
            }
            let mut loci_counter: usize = 0;
            let step: usize = length / loci;
            for j in (0..length).step_by(step) {
                loci_counter += 1;
                if loci_counter > loci {
                    break;
                }
                let n_alleles: usize = rng.random_range(1..=total_alleles);
                let allele_ids: Vec<usize> = (0..n_alleles)
                    .map(|_| rng.random_range(0..total_alleles))
                    .collect();
                let locus_length: usize = allele_ids
                    .iter()
                    .map(|&i| alleles.sequences[i].len())
                    .max()
                    .unwrap_or(1);
                let (start, end) = if (j + locus_length) < chromosomes.lengths[i] {
                    (j, j + locus_length)
                } else {
                    (
                        chromosomes.lengths[i] - locus_length,
                        chromosomes.lengths[i],
                    )
                };
                out_loci.push(Self {
                    chromosome_id: i,
                    start,
                    end,
                    allele_ids: allele_ids.clone(),
                });
                for a in allele_ids {
                    out_loci_alleles.push(LocusAllele {
                        locus_id: out_loci.len() - 1,
                        allele_id: a,
                    });
                }
            }
        }
        Ok((out_loci, out_loci_alleles))
    }
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

impl Genome {
    /// Constructs a fully validated `Genome` containing chromosomes, allele
    /// definitions, physical loci, and the flattened locus‑allele relational map.
    ///
    /// # Overview
    ///
    /// This initializer composes the three foundational genomic structures:
    ///
    /// - [`Chromosomes`]: physical linkage groups with validated lengths and
    ///   chromosome‑specific LD decay distances.
    /// - [`Alleles`]: global registry of unique allele sequences.
    /// - [`Locus`]: physical loci placed proportionally across chromosomes,
    ///   each with a random multi‑allelic state determined by `seed`.
    ///
    /// The resulting `Genome` acts as the CPU‑side blueprint for downstream
    /// GPU genotype tensor construction, founder haplotype generation,
    /// recombination simulation, and breeding‑value computation.
    ///
    /// # Linkage Disequilibrium Model
    ///
    /// Each chromosome is assigned a characteristic LD decay distance `L`
    /// (in base pairs). Simmy assumes the exponential LD decay model:
    ///
    /// ```text
    /// r²(d) = r²(0) exp(-d / L)
    /// ```
    ///
    /// where:
    ///
    /// - `r²(d)` is the expected linkage disequilibrium between two loci
    ///   separated by distance `d`.
    /// - `r²(0)` is assumed to be `1.0`.
    /// - `d` is the physical distance between loci in base pairs.
    /// - `L` is the chromosome‑specific LD decay distance.
    ///
    /// Consequently:
    ///
    /// ```text
    /// d = L
    /// ```
    ///
    /// implies:
    ///
    /// ```text
    /// r²(L) = exp(-1) ≈ 0.368
    /// ```
    ///
    /// Larger LD decay distances imply longer haplotype blocks and slower
    /// LD decay. Smaller LD decay distances imply weaker long‑range linkage
    /// and more rapid LD decay.
    ///
    /// # Determinism
    ///
    /// Locus generation is fully deterministic under the supplied `seed`.
    /// All allele sampling and locus coordinate placement are reproducible.
    ///
    /// # Parameters
    ///
    /// - `n_chromosomes`: Number of chromosomes to construct.
    /// - `chromosome_lengths`: Optional slice of chromosome lengths in base pairs.
    ///   If `None`, all chromosomes default to `1_000_000` bp.
    /// - `ld_decay_distances`: Optional slice of chromosome‑specific LD decay
    ///   distances. If `None`, all chromosomes default to `2_000` bp.
    /// - `n_max_alleles`: Number of unique allele sequences to generate or import.
    /// - `allele_sequences`: Optional slice of allele names. If `None`, names
    ///   are generated using mixed‑radix SNP encoding.
    /// - `n_loci`: Total number of loci to distribute proportionally across chromosomes.
    /// - `seed`: RNG seed controlling allele sampling and locus placement.
    ///
    /// # Returns
    ///
    /// A fully validated `Genome` containing:
    ///
    /// - `chromosomes`: physical chromosomes with chromosome lengths and
    ///   LD decay parameters.
    /// - `alleles`: global allele registry.
    /// - `loci`: physical loci with chromosome assignments, coordinates,
    ///   and valid allele states.
    /// - `loci_alleles`: flattened `(locus_id, allele_id)` mapping suitable
    ///   for GPU genotype tensor construction.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - Chromosome lengths do not match `n_chromosomes`.
    /// - LD decay distances do not match `n_chromosomes`.
    /// - Any LD decay distance is zero.
    /// - Allele sequence count does not match `n_max_alleles`.
    /// - Duplicate allele sequences are detected.
    /// - The genome is too small to place `n_loci` loci of maximum allele width.
    pub fn new(
        n_chromosomes: usize,
        chromosome_lengths: Option<&[usize]>,
        ld_decay_distances: Option<&[usize]>,
        n_max_alleles: usize,
        allele_sequences: Option<&[&str]>,
        n_loci: usize,
        seed: u64,
    ) -> Result<Self> {
        let chromosomes = Chromosomes::new(n_chromosomes, chromosome_lengths, ld_decay_distances)?;
        let alleles = Alleles::new(n_max_alleles, allele_sequences)?;
        let (loci, loci_alleles) = Locus::new(&chromosomes, &alleles, n_loci, seed)?;
        Ok(Self {
            chromosomes,
            alleles,
            loci,
            loci_alleles,
        })
    }
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
    pub populations: Vec<String>,
    /// User-defined categorization, breeding tiers, or selection groups.
    pub classifications: Vec<String>,
    /// Arbitrary historical logs, pedigree descriptions, or metadata notes.
    pub notes: Vec<String>,
}

impl Entries {
    /// Constructs an `Entries` struct containing demographic metadata for `n` individuals.
    ///
    /// # Overview
    /// This constructor builds five parallel metadata vectors:
    /// - `names`
    /// - `species`
    /// - `populations`
    /// - `classifications`
    /// - `notes`
    ///
    /// Each vector is guaranteed to have length `n`, ensuring strict struct‑of‑arrays
    /// (SoA) alignment. When optional slices are provided, they are copied verbatim
    /// into owned `Vec<String>` values.
    ///
    /// When a field is `None`, a deterministic, zero‑padded default is generated.
    /// The padding width is derived from `n - 1`, ensuring consistent formatting
    /// across all identifiers. For example:
    ///
    /// - `n = 3`  → `"entry_0"`, `"entry_1"`, `"entry_2"`
    /// - `n = 12` → `"entry_00"`, `"entry_01"`, ..., `"entry_11"`
    ///
    /// The same padding rule applies to:
    /// - `names`: `"entry_<i>"`
    /// - `species`: `"species_<i>"`
    /// - `populations`: `"population_<i>"`
    /// - `classifications`: `"classification_<i>"`
    ///
    /// Notes default to empty strings.
    ///
    /// These defaults provide meaningful semantic labels for debugging, cohort slicing,
    /// and metadata inspection while maintaining deterministic reproducibility.
    ///
    /// # Determinism
    /// All default values are generated deterministically based on `n`. This ensures
    /// reproducible cohort construction and stable indexing across simulation runs.
    ///
    /// # Parameters
    /// - `n`: Number of entries to construct. Must be non‑zero.
    /// - `names`: Optional slice of entry names.
    /// - `species`: Optional slice of species identifiers.
    /// - `populations`: Optional slice of population or cohort identifiers.
    /// - `classifications`: Optional slice of breeding tier or category labels.
    /// - `notes`: Optional slice of free‑form metadata notes.
    ///
    /// # Returns
    /// A fully validated `Entries` struct with all five metadata vectors aligned by index.
    ///
    /// # Validation
    /// - Ensures `n > 0`.
    /// - Ensures each provided slice has length `n`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `n == 0`,
    /// - Any provided slice does not have length `n`.
    pub fn new(
        n: usize,
        names: Option<&[&str]>,
        species: Option<&[&str]>,
        populations: Option<&[&str]>,
        classifications: Option<&[&str]>,
        notes: Option<&[&str]>,
    ) -> Result<Self> {
        ensure!(n > 0, "Number of entries need to non-zero!");
        let n_digits: usize = format!("{}", n - 1).len();
        let names: Vec<String> = match names {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => (0..n).map(|x| format!("entry_{:0>n_digits$}", x)).collect(),
        };
        let species: Vec<String> = match species {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => (0..n)
                .map(|x| format!("species_{:0>n_digits$}", x))
                .collect(),
        };
        let populations: Vec<String> = match populations {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => (0..n)
                .map(|x| format!("population_{:0>n_digits$}", x))
                .collect(),
        };
        let classifications: Vec<String> = match classifications {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => (0..n)
                .map(|x| format!("classification_{:0>n_digits$}", x))
                .collect(),
        };
        let notes: Vec<String> = match notes {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => vec!["".to_owned(); n],
        };
        ensure!(
            n == names.len(),
            "The expected number of entries (n={}) does not match the names (names.len()={})!",
            n,
            names.len()
        );
        ensure!(
            n == species.len(),
            "The expected number of entries (n={}) does not match the species (species.len()={})!",
            n,
            species.len()
        );
        ensure!(
            n == populations.len(),
            "The expected number of entries (n={}) does not match the populations (populations.len()={})!",
            n,
            populations.len()
        );
        ensure!(
            n == classifications.len(),
            "The expected number of entries (n={}) does not match the classifications (classifications.len()={})!",
            n,
            classifications.len()
        );
        ensure!(
            n == notes.len(),
            "The expected number of entries (n={}) does not match the notes (notes.len()={})!",
            n,
            notes.len()
        );
        Ok(Self {
            names,
            species,
            populations,
            classifications,
            notes,
        })
    }
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

impl Traits {
    /// Constructs a `Traits` struct containing metadata for `n` quantitative traits.
    ///
    /// # Overview
    /// This constructor builds two parallel metadata vectors:
    /// - `names`
    /// - `notes`
    ///
    /// Each vector is guaranteed to have length `n`, ensuring strict struct‑of‑arrays
    /// (SoA) alignment. When optional slices are provided, they are copied verbatim
    /// into owned `Vec<String>` values.
    ///
    /// When `names` is `None`, deterministic zero‑padded identifiers are generated.
    /// The padding width is derived from `n - 1`, ensuring consistent formatting
    /// across all trait names. For example:
    ///
    /// - `n = 3`  → `"trait_0"`, `"trait_1"`, `"trait_2"`
    /// - `n = 12` → `"trait_00"`, `"trait_01"`, ..., `"trait_11"`
    ///
    /// Notes default to empty strings.
    ///
    /// These defaults provide meaningful semantic labels for debugging, trait indexing,
    /// and multi‑trait selection workflows while maintaining deterministic reproducibility.
    ///
    /// # Determinism
    /// All default values are generated deterministically based on `n`. This ensures
    /// reproducible trait‑set construction and stable indexing across simulation runs.
    ///
    /// # Parameters
    /// - `n`: Number of traits to construct. Must be non‑zero.
    /// - `names`: Optional slice of trait names.
    /// - `notes`: Optional slice of trait descriptions or metadata.
    ///
    /// # Returns
    /// A fully validated `Traits` struct with `names` and `notes` aligned by index.
    ///
    /// # Validation
    /// - Ensures `n > 0`.
    /// - Ensures each provided slice has length `n`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `n == 0`,
    /// - `names` or `notes` is provided with a length different from `n`.
    pub fn new(n: usize, names: Option<&[&str]>, notes: Option<&[&str]>) -> Result<Self> {
        ensure!(n > 0, "Number of traits need to non-zero!");
        let n_digits: usize = format!("{}", n - 1).len();
        let names: Vec<String> = match names {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => (0..n).map(|x| format!("trait_{:0>n_digits$}", x)).collect(),
        };
        let notes: Vec<String> = match notes {
            Some(x) => x.iter().map(|&x| x.to_owned()).collect(),
            None => vec!["".to_owned(); n],
        };
        ensure!(
            n == names.len(),
            "The expected number of traits (n={}) does not match the names (names.len()={})!",
            n,
            names.len()
        );
        ensure!(
            n == notes.len(),
            "The expected number of traits (n={}) does not match the notes (notes.len()={})!",
            n,
            notes.len()
        );
        Ok(Self { names, notes })
    }
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


impl GenotypeData {
    pub fn founders(ctx: &GpuContext, genome: &Genome, founder_entries: &Entries, af_shape: f32, seed: u64) -> Result<Self> {
        let n: usize = founder_entries.names.len();
        let p: usize = genome.loci_alleles.len();
        ensure!(n > 0, "Number of founders need to non-zero!");
        ensure!(
            p > 0,
            "Number of loci-alleles need to non-zero!"
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let beta = Beta::new(af_shape, af_shape).context("Failed to initialize Beta distribution: parameters must be greater than 0")?;
        let mut data_tmp: Vec<f32> = Vec::with_capacity(n*p);
        for _ in 0..(n*p) {
            data_tmp.push(beta.sample(&mut rng));
        }
        let data = GpuTensor::from_f32(ctx, &data_tmp, &[n as u32, p as u32], None, None)?;
        Ok(Self { entry_ids: (0..n).collect(), locus_allele_ids: (0..p).collect(), data })
    }
    pub fn new() {
        // Account for LD decay here to generate a population from the founder genotypes...
        todo!()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    //////////////////////////////
    // Chromosomes::new tests
    //////////////////////////////
    #[test]
    fn chromosomes_default_lengths() -> Result<()> {
        let chr = Chromosomes::new(3, None, None)?;
        assert_eq!(chr.chromosomes, vec!["chr_0", "chr_1", "chr_2"]);
        assert_eq!(chr.lengths, vec![1_000_000, 1_000_000, 1_000_000]);
        Ok(())
    }
    #[test]
    fn chromosomes_custom_lengths() -> Result<()> {
        let chr = Chromosomes::new(3, Some(&[10, 20, 30]), None)?;
        assert_eq!(chr.chromosomes, &["chr_0", "chr_1", "chr_2"]);
        assert_eq!(chr.lengths, &[10, 20, 30]);
        Ok(())
    }
    #[test]
    fn chromosomes_length_mismatch_fails() {
        let result = Chromosomes::new(3, Some(&[10, 20]), None);
        assert!(result.is_err());
    }
    #[test]
    fn chromosomes_zero_n() -> Result<()> {
        let result = Chromosomes::new(0, None, None);
        assert!(result.is_err());
        Ok(())
    }
    //////////////////////////////
    // Alleles::new tests
    //////////////////////////////
    #[test]
    fn alleles_default_n_leq_5() -> Result<()> {
        let a = Alleles::new(5, None)?;
        assert_eq!(a.sequences, &["A", "T", "C", "G", "D"]);
        Ok(())
    }
    #[test]
    fn alleles_default_n_10() -> Result<()> {
        let a = Alleles::new(10, None)?;
        assert_eq!(
            a.sequences,
            &["A", "T", "C", "G", "D", "TA", "TT", "TC", "TG", "TD"]
        );
        Ok(())
    }
    #[test]
    fn alleles_default_n_20() -> Result<()> {
        let a = Alleles::new(20, None)?;
        // Check first 5 and last 5 only
        assert_eq!(&a.sequences[..5], &["A", "T", "C", "G", "D"]);
        assert_eq!(&a.sequences[15..20], &["GA", "GT", "GC", "GG", "GD"]);
        Ok(())
    }
    #[test]
    fn alleles_custom_names() -> Result<()> {
        let custom = &["X", "Y", "Z"];
        let a = Alleles::new(3, Some(custom))?;
        assert_eq!(a.sequences, custom);
        Ok(())
    }
    #[test]
    fn alleles_custom_name_length_mismatch_fails() {
        let result = Alleles::new(3, Some(&["A", "B"]));
        assert!(result.is_err());
    }
    #[test]
    fn alleles_no_duplicates() -> Result<()> {
        let a = Alleles::new(50, None)?;
        let mut sorted = a.sequences.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), a.sequences.len());
        Ok(())
    }
    #[test]
    fn alleles_large_n() -> Result<()> {
        let a = Alleles::new(500, None)?;
        assert_eq!(a.sequences.len(), 500);
        // Check that the last allele is multi-character
        assert!(a.sequences[499].len() >= 3);
        Ok(())
    }
    //////////////////////////////
    // Locus and LocusAllele
    //////////////////////////////
    #[test]
    fn test_locus_new_basic() {
        let chromosomes = Chromosomes::new(3, Some(&[100, 200, 300]), None).unwrap();
        let alleles = Alleles::new(5, None).unwrap();
        let (loci, locus_alleles) = Locus::new(&chromosomes, &alleles, 10, 42).unwrap();
        println!("loci: {:?}", loci);
        assert_eq!(loci.len(), 10);
        for locus in &loci {
            assert!(!locus.allele_ids.is_empty());
            assert!(locus.chromosome_id < chromosomes.chromosomes.len());
        }
        let expected_count: usize = loci.iter().map(|l| l.allele_ids.len()).sum();
        assert_eq!(locus_alleles.len(), expected_count);
    }
    #[test]
    fn test_locus_coordinates_and_width() {
        let chromosomes = Chromosomes::new(1, Some(&[50]), None).unwrap();
        let alleles = Alleles::new(5, None).unwrap();
        let (loci, _) = Locus::new(&chromosomes, &alleles, 5, 42).unwrap();
        for locus in &loci {
            assert!(locus.start < 50);
            assert!(locus.end <= 50);
            assert!(locus.start < locus.end);
            let expected_len = locus
                .allele_ids
                .iter()
                .map(|&i| alleles.sequences[i].len())
                .max()
                .unwrap();

            assert_eq!(locus.end - locus.start, expected_len);
        }
    }
    #[test]
    fn test_locus_allele_mapping_correctness() {
        let chromosomes = Chromosomes::new(2, Some(&[100, 100]), None).unwrap();
        let alleles = Alleles::new(4, None).unwrap();
        let (loci, locus_alleles) = Locus::new(&chromosomes, &alleles, 6, 42).unwrap();
        let expected_count: usize = loci.iter().map(|l| l.allele_ids.len()).sum();
        assert_eq!(locus_alleles.len(), expected_count);
        for la in &locus_alleles {
            assert!(la.locus_id < loci.len());
            assert!(la.allele_id < alleles.sequences.len());
        }
        for (locus_id, locus) in loci.iter().enumerate() {
            let mapped: Vec<usize> = locus_alleles
                .iter()
                .filter(|la| la.locus_id == locus_id)
                .map(|la| la.allele_id)
                .collect();
            assert_eq!(mapped.len(), locus.allele_ids.len());
            for a in &mapped {
                assert!(locus.allele_ids.contains(a));
            }
        }
    }
    //////////////////////////////
    // Genome
    //////////////////////////////
    #[test]
    fn test_genome_new_basic() {
        // Simple 3‑chromosome genome
        let n_chromosomes = 3;
        let chromosome_lengths = vec![100usize, 200, 300];
        // 5 alleles using automatic SNP‑based generation
        let n_max_alleles = 5;
        let allele_sequences = None;
        // 10 loci distributed proportionally
        let n_loci = 10;
        // Deterministic seed
        let seed = 12345;
        let genome = Genome::new(
            n_chromosomes,
            Some(&chromosome_lengths),
            None,
            n_max_alleles,
            allele_sequences,
            n_loci,
            seed,
        )
        .unwrap();
        // Chromosome validation
        assert_eq!(genome.chromosomes.chromosomes.len(), 3);
        assert_eq!(genome.chromosomes.lengths, vec![100, 200, 300]);
        // Allele validation
        assert_eq!(genome.alleles.sequences.len(), 5);
        // Locus count must match n_loci
        assert_eq!(genome.loci.len(), n_loci);
        // All loci must reference valid chromosomes
        for locus in &genome.loci {
            assert!(locus.chromosome_id < n_chromosomes);
        }
        // Coordinates must be valid and clamped
        for locus in &genome.loci {
            let chr_len = genome.chromosomes.lengths[locus.chromosome_id];
            assert!(locus.start < chr_len);
            assert!(locus.end <= chr_len);
            assert!(locus.start < locus.end);
        }
        // Flattened mapping must match sum of allele counts
        let expected_flat_count: usize = genome.loci.iter().map(|l| l.allele_ids.len()).sum();
        assert_eq!(genome.loci_alleles.len(), expected_flat_count);
        // Mapping must be consistent
        for la in &genome.loci_alleles {
            assert!(la.locus_id < genome.loci.len());
            assert!(la.allele_id < genome.alleles.sequences.len());
            assert!(genome.loci[la.locus_id].allele_ids.contains(&la.allele_id));
        }
    }
    #[test]
    fn test_genome_invariants() {
        let genome = Genome::new(4, Some(&[150, 250, 350, 450]), None, 12, None, 20, 999).unwrap();
        // Invariant 1: All loci reference valid chromosomes
        for locus in &genome.loci {
            assert!(locus.chromosome_id < genome.chromosomes.chromosomes.len());
        }
        // Invariant 2: No locus has zero alleles
        for locus in &genome.loci {
            assert!(!locus.allele_ids.is_empty());
        }
        // Invariant 3: No locus exceeds chromosome boundaries
        for locus in &genome.loci {
            let chr_len = genome.chromosomes.lengths[locus.chromosome_id];
            assert!(locus.start < chr_len);
            assert!(locus.end <= chr_len);
        }
        // Invariant 4: Flattened mapping is complete
        let expected_flat: usize = genome.loci.iter().map(|l| l.allele_ids.len()).sum();
        assert_eq!(genome.loci_alleles.len(), expected_flat);
    }
    #[test]
    fn test_genome_regression_snapshot() {
        let genome = Genome::new(3, Some(&[100, 200, 300]), None, 5, None, 10, 42).unwrap();
        // Snapshot: chromosome lengths
        assert_eq!(genome.chromosomes.lengths, vec![100, 200, 300]);
        // Snapshot: first locus
        let first = &genome.loci[0];
        assert_eq!(first.chromosome_id, 0);
        assert_eq!(first.start, 0);
        assert_eq!(
            first.end,
            first
                .allele_ids
                .iter()
                .map(|&i| genome.alleles.sequences[i].len())
                .max()
                .unwrap()
        );
        // Snapshot: first locus allele IDs
        assert_eq!(first.allele_ids, vec![3, 0]); // deterministic under seed 12345
    }
    //////////////////////////////
    // Entries
    //////////////////////////////
    #[test]
    fn test_entries_new_defaults() {
        let entries = Entries::new(11, None, None, None, None, None).unwrap();
        // n = 11 → n_digits = len("10") = 2 → 00, 01, ..., 10
        assert_eq!(
            entries.names,
            vec![
                "entry_00", "entry_01", "entry_02", "entry_03", "entry_04", "entry_05", "entry_06",
                "entry_07", "entry_08", "entry_09", "entry_10"
            ]
        );
        assert_eq!(
            entries.species,
            vec![
                "species_00",
                "species_01",
                "species_02",
                "species_03",
                "species_04",
                "species_05",
                "species_06",
                "species_07",
                "species_08",
                "species_09",
                "species_10"
            ]
        );
        assert_eq!(
            entries.populations,
            vec![
                "population_00",
                "population_01",
                "population_02",
                "population_03",
                "population_04",
                "population_05",
                "population_06",
                "population_07",
                "population_08",
                "population_09",
                "population_10"
            ]
        );
        assert_eq!(
            entries.classifications,
            vec![
                "classification_00",
                "classification_01",
                "classification_02",
                "classification_03",
                "classification_04",
                "classification_05",
                "classification_06",
                "classification_07",
                "classification_08",
                "classification_09",
                "classification_10"
            ]
        );
        assert_eq!(entries.notes, vec![""; 11]);
    }
    #[test]
    fn test_entries_new_with_provided_fields() {
        let names = ["A", "B", "C"];
        let species = ["dog", "cat", "horse"];
        let populations = ["founder", "F1", "F2"];
        let classifications = ["elite", "candidate", "discard"];
        let notes = ["note1", "note2", "note3"];
        let entries = Entries::new(
            3,
            Some(&names),
            Some(&species),
            Some(&populations),
            Some(&classifications),
            Some(&notes),
        )
        .unwrap();
        assert_eq!(entries.names, vec!["A", "B", "C"]);
        assert_eq!(entries.species, vec!["dog", "cat", "horse"]);
        assert_eq!(entries.populations, vec!["founder", "F1", "F2"]);
        assert_eq!(
            entries.classifications,
            vec!["elite", "candidate", "discard"]
        );
        assert_eq!(entries.notes, vec!["note1", "note2", "note3"]);
    }
    #[test]
    fn test_entries_new_mixed_fields() {
        let names = ["X", "Y"];
        let notes = ["alpha", "beta"];
        let entries = Entries::new(2, Some(&names), None, None, None, Some(&notes)).unwrap();
        assert_eq!(entries.names, vec!["X", "Y"]);
        assert_eq!(entries.species, vec!["species_0", "species_1"]);
        assert_eq!(entries.populations, vec!["population_0", "population_1"]);
        assert_eq!(
            entries.classifications,
            vec!["classification_0", "classification_1"]
        );
        assert_eq!(entries.notes, vec!["alpha", "beta"]);
    }
    #[test]
    fn test_entries_new_length_mismatch() {
        let names = ["A", "B", "C"]; // length 3
        let err = Entries::new(
            2,            // n = 2
            Some(&names), // length mismatch
            None,
            None,
            None,
            None,
        );
        assert!(err.is_err());
    }
    #[test]
    fn test_entries_new_default_formatting() {
        let entries = Entries::new(12, None, None, None, None, None).unwrap();
        // 12 → n_digits = 2
        assert_eq!(entries.names[0], "entry_00");
        assert_eq!(entries.names[11], "entry_11");
        assert_eq!(entries.species[11], "species_11");
        assert_eq!(entries.populations[11], "population_11");
        assert_eq!(entries.classifications[11], "classification_11");
    }
    //////////////////////////////
    // Traits
    //////////////////////////////
    #[test]
    fn test_traits_new_defaults() {
        // n = 11 → n_digits = len("10") = 2 → 00..10
        let traits = Traits::new(11, None, None).unwrap();
        assert_eq!(
            traits.names,
            vec![
                "trait_00", "trait_01", "trait_02", "trait_03", "trait_04", "trait_05", "trait_06",
                "trait_07", "trait_08", "trait_09", "trait_10"
            ]
        );
        assert_eq!(traits.notes, vec![""; 11]);
    }
    #[test]
    fn test_traits_new_with_provided_fields() {
        let names = ["yield", "height", "protein"];
        let notes = ["kg/ha", "cm", "percent"];
        let traits = Traits::new(3, Some(&names), Some(&notes)).unwrap();
        assert_eq!(traits.names, vec!["yield", "height", "protein"]);
        assert_eq!(traits.notes, vec!["kg/ha", "cm", "percent"]);
    }
    #[test]
    fn test_traits_new_mixed_fields() {
        let names = ["traitA", "traitB"];
        let traits = Traits::new(2, Some(&names), None).unwrap();
        assert_eq!(traits.names, vec!["traitA", "traitB"]);
        assert_eq!(traits.notes, vec!["", ""]);
    }
    #[test]
    fn test_traits_new_length_mismatch() {
        let names = ["A", "B", "C"]; // length 3
        let result = Traits::new(2, Some(&names), None);
        assert!(result.is_err());
    }
    #[test]
    fn test_traits_new_zero_n() {
        let result = Traits::new(0, None, None);
        assert!(result.is_err());
    }
    #[test]
    fn test_traits_new_padding_width() {
        // n = 100 → n_digits = len("99") = 2 → trait_00..trait_99
        let traits = Traits::new(100, None, None).unwrap();
        assert_eq!(traits.names[0], "trait_00");
        assert_eq!(traits.names[9], "trait_09");
        assert_eq!(traits.names[10], "trait_10");
        assert_eq!(traits.names[99], "trait_99");
    }
}
