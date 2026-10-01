struct Params {
    n_loci_alleles: u32,
    n_loci: u32,
    seed: u32,
}

struct MatingPair {
    p1_idx: u32,
    p2_idx: u32,
    sex_p1: u32,
    sex_p2: u32,
}

struct LocusData {
    start_col: u32,
    end_col: u32,
    r: f32,
    metadata: u32,
}

@group(0) @binding(0) var<storage, read> parent_genotypes: array<f32>;
@group(0) @binding(1) var<storage, read_write> offspring_genotypes: array<f32>;
@group(0) @binding(2) var<storage, read> mating_pairs: array<MatingPair>;
@group(0) @binding(3) var<storage, read> locus_data: array<LocusData>;
@group(0) @binding(4) var<uniform> params: Params;

var<private> prng_state: u32;

fn pcg_random() -> f32 {
    let state = prng_state;
    prng_state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let u_val = (word >> 22u) ^ word;
    return f32(u_val) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let off_idx = global_id.x;
    if (off_idx >= arrayLength(&mating_pairs)) {
        return;
    }

    let pair = mating_pairs[off_idx];
    
    prng_state = params.seed ^ (off_idx * 19937u);

    var homolog_p1 = u32(pcg_random() < 0.5);
    var homolog_p2 = u32(pcg_random() < 0.5);

    let n_loci_alleles = params.n_loci_alleles;

    for (var i = 0u; i < params.n_loci; i++) {
        let loc = locus_data[i];
        let is_sex_chrom = (loc.metadata & 1u) != 0u; // Extract Bit 0: Evaluate to true if this locus resides on a sex chromosome
        let is_new_chrom = (loc.metadata & 2u) != 0u; // Extract Bit 1: Evaluate to true if this locus is the start of a new chromosome (which will bypass linkage and force independent assortment)

        var r_p1 = loc.r;
        var r_p2 = loc.r;

        // Suppress recombination in heterogametic sex chromosomes
        if (is_sex_chrom && !is_new_chrom) {
            if (pair.sex_p1 == 2u) { r_p1 = 1.0; }
            if (pair.sex_p2 == 2u) { r_p2 = 1.0; }
        }

        // Evaluate crossover probability for both parents independently.
        // `pcg_random()` generates a float between 0.0 and 1.0.
        // `r` is the linkage probability (ranging from 0.5 to 1.0):
        //   - If r = 1.0 (complete linkage), pcg_random() > 1.0 is never true --> no crossover.
        //   - If r = 0.5 (unlinked/new chromosome), 50% chance to crossover --> independent assortment.
        // When a crossover occurs, we flip the active homologous chromosome from 0 to 1 (or 1 to 0)
        // using the math `1u - homolog`.
        if (pcg_random() > r_p1) { homolog_p1 = 1u - homolog_p1; }
        if (pcg_random() > r_p2) { homolog_p2 = 1u - homolog_p2; }

        for (var a = loc.start_col; a < loc.end_col; a++) {
            let src_idx_p1 = pair.p1_idx * n_loci_alleles * 2u + a * 2u + homolog_p1;
            let dst_idx_0 = off_idx * n_loci_alleles * 2u + a * 2u + 0u;
            offspring_genotypes[dst_idx_0] = parent_genotypes[src_idx_p1];

            let src_idx_p2 = pair.p2_idx * n_loci_alleles * 2u + a * 2u + homolog_p2;
            let dst_idx_1 = off_idx * n_loci_alleles * 2u + a * 2u + 1u;
            offspring_genotypes[dst_idx_1] = parent_genotypes[src_idx_p2];
        }
    }
}