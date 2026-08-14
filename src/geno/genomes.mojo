@fieldwise_init
struct Genomes:
    var chromosomes: List[String]
    var lengths_per_chromosome: List[UInt]
    var alleles: List[String]
    var loci: List[Float64]
    var entries: List[String]
    var allele_frequencies: List[List[Float64]]
