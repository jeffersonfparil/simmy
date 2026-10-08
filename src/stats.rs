use crate::entry::Entry;
use crate::genotype::{Chromosome, Locus};
use crate::{
    dataset::Dataset,
    linalg::{context::GpuContext, kernel::GpuKernel, tensor::GpuTensor},
};
use anyhow::{Result, ensure};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    n_entries: usize,
    ploidy: usize,
    with_sex: bool,
    genome_size: usize,
    n_chromosomes: usize,
    n_loci: usize,
    n_loci_alleles: usize,
    n_alleles_per_locus_min: usize,
    n_alleles_per_locus_mean: usize,
    n_alleles_per_locus_max: usize,
    // TODO: per species per group:
    //  - genotype per window per chromosome:
    //      + allele frequency spectrum
    //      + LD map
    //      + ...?
    //  - inbreeding rate
    //  - phenotypes per trait:
    //      + mean, sd, min, max, median
    //      + distributions
    //      + correlations
    //      + ...?
}

impl fmt::Display for DatasetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "------------------------")?;
        writeln!(f, "DatasetStats struct")?;
        writeln!(f, "------------------------")?;
        writeln!(f, "entries: {}", self.n_entries)?;
        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct AlleleFrequencyPerLocusPerAllele {
    allele_freqs: Vec<f32>,        // frequency of each allele per locus
    homozygosity_zeroes: Vec<f32>, // frequency that each allele is fixed at 0.00 per locus
    homozygosity_ones: Vec<f32>,   // frequency that each allele is fixed at 1.00 per locus
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct AlleleFrequencyPerLocus {
    // Note that loci here are unsorted and maps directly to Dataset.loci vector!
    n_alleles: Vec<usize>,        // number of alleles per locus
    major_allele_freqs: Vec<f32>, // max
    minor_allele_freqs: Vec<f32>, // min
    mu_allele_freqs: Vec<f32>,    // mean
    sd_allele_freqs: Vec<f32>,    // sd
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct AlleleFrequencyPerChromosome {
    genome: Vec<Chromosome>, // decided to store these for self-sufficiency as these map as consecutive elements to the Vec<usize> and Vec<f32> below  --> hoping they are not too big...
    window_bp: usize,        // window size in bases
    positions: Vec<Vec<(usize, usize)>>, // start and end positions of each window per chromosome
    n_loci: Vec<Vec<usize>>, // loci counts per window per chromosome
    mu: Vec<Vec<f32>>,       // mean allele frequencies per window per chromosome
    sd: Vec<Vec<f32>>,       // standard deviation of allele frequencies per window per chromosome
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticDiversityPerLocus {
    n_alleles: Vec<usize>,           // number of alleles per locus
    homozygosity_ones_sum: Vec<f32>, // sum of frequency of fixed alleles at 1.00 across all alleles per locus
    homozygosity_ones_max: Vec<f32>, // max frequency of fixed allele at 1.00 across all alleles per locus
    heterozygosity: Vec<f32>, // heterozygosity = 1.00 - homozygosity_ones_sum, i.e. closer to the expected heterozygosity present in most tools (He = 1 - Σ p²), because we actually measure the proportion of entries at a locus that are not fixed to 1.00 at any of the alleles!
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticDiversityPerChromosome {
    genome: Vec<Chromosome>, // decided to store these for self-sufficiency as these map as consecutive elements to the Vec<usize> and Vec<f32> below  --> hoping they are not too big...
    window_bp: usize,        // window size in bases
    positions: Vec<Vec<(usize, usize)>>, // start and end positions of each window per chromosome
    n_loci: Vec<Vec<usize>>, // loci counts per window per chromosome
    heterozygosity_mu: Vec<Vec<f32>>, // mean heterozygosities per window per chromosome
    heterozygosity_sd: Vec<Vec<f32>>, // standard deviation of heterozygosities per window per chromosome
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticDiversityPerEntry {
    entries: Vec<Entry>,
    n_loci: usize,
    homozygosity: Vec<f32>,   // frequency of fixed loci per entry
    heterozygosity: Vec<f32>, // frequency of heterozygous loci per entry (heterozygosity = 1.00 - homozygosity_ones_sum)
}

// TODO: put these into src/stat/ and divide them into something like: stat.rs with mod etc..., allele_freq.rs, genomic_diversity.rs, and phenomic_diversity.rs
impl Dataset {
    pub fn sortperm_entries(&self) -> Result<Vec<usize>> {
        self.check()?;
        let n: usize = self.entries.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| self.entries[i].cmp(&self.entries[j]));
        Ok(idx)
    }
    pub fn sortperm_loci(&self) -> Result<Vec<usize>> {
        self.check()?;
        let n: usize = self.loci.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| self.loci[i].cmp(&self.loci[j]));
        Ok(idx)
    }
    pub fn allele_freq_per_locus_allele(
        &self,
        ctx: &GpuContext,
    ) -> Result<AlleleFrequencyPerLocusPerAllele> {
        self.check()?;
        let n_entries: usize = self.entries.len();
        let n_loci_alleles: usize = self.genotype_data.shape[1] as usize;
        let kernel: GpuKernel = GpuKernel { ctx };
        let allele_dosages: GpuTensor = kernel
            .add(
                &self
                    .genotype_data
                    .slice_view(&[(0, n_entries), (0, n_loci_alleles), (0, 1)])?,
                &self
                    .genotype_data
                    .slice_view(&[(0, n_entries), (0, n_loci_alleles), (1, 2)])?,
            )?
            .transpose_view(None)?; // n_loci_alleles x n_entries
        let ones: GpuTensor = GpuTensor::from_vec_f32(
            ctx,
            &vec![1.0; n_entries],
            &[n_entries as u32, 1],
            None,
            None,
        )?;
        let allele_freqs: Vec<f32> = kernel
            .div_scalar_matrix(
                &kernel.matmul(&allele_dosages, &ones)?,
                (n_entries * self.ploidy) as f32,
            )?
            .to_vec_f32(ctx)?; // n_loci_alleles
        let homozygosity_zeroes: Vec<f32> = kernel
            .div_scalar_matrix(
                &kernel.matmul(&kernel.eq_scalar(&allele_dosages, 0.0)?, &ones)?,
                n_entries as f32,
            )?
            .to_vec_f32(ctx)?;
        let homozygosity_ones: Vec<f32> = kernel
            .div_scalar_matrix(
                &kernel.matmul(
                    &kernel.eq_scalar(&allele_dosages, self.ploidy as f32)?,
                    &ones,
                )?,
                n_entries as f32,
            )?
            .to_vec_f32(ctx)?;
        // Output
        Ok(AlleleFrequencyPerLocusPerAllele {
            allele_freqs,
            homozygosity_zeroes,
            homozygosity_ones,
        })
    }
    pub fn allele_freq_per_locus(
        &self,
        allele_freqs: &AlleleFrequencyPerLocusPerAllele,
    ) -> Result<AlleleFrequencyPerLocus> {
        self.check()?;
        let n_loci_alleles: usize = self.genotype_data.shape[1] as usize;
        ensure!(
            allele_freqs.allele_freqs.len() == n_loci_alleles,
            "The dataset and input allele frequencies do not match!"
        );
        let n_loci: usize = self.loci.len();
        let mut n_alleles: Vec<usize> = Vec::with_capacity(n_loci); // allele counts per locus
        let mut major_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci); // max
        let mut minor_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci); // min
        let mut mu_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci); // mean
        let mut sd_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci); // sd
        for locus in &self.loci {
            let mut p: f32 = 0.0;
            let mut q: f32 = 1.0;
            let mut fs: Vec<f32> = Vec::with_capacity(locus.col_idx.len());
            for &j in &locus.col_idx {
                let f: f32 = allele_freqs.allele_freqs[j];
                ensure!(
                    (0.0..=1.0).contains(&f),
                    "Allele dosages are not consistent with the expected ploidy of the dataset (see locus: {:?})!",
                    locus
                );
                p = p.max(f);
                q = q.min(f);
                fs.push(f);
            }
            let n: usize = fs.len();
            let mu: f32 = fs.iter().sum::<f32>() / (n as f32);
            let sd: f32 = (fs.iter().map(|&x| (x - mu).powi(2)).sum::<f32>() / (n as f32)).sqrt();
            n_alleles.push(n);
            major_allele_freqs.push(p);
            minor_allele_freqs.push(q);
            mu_allele_freqs.push(mu);
            sd_allele_freqs.push(sd);
        }
        Ok(AlleleFrequencyPerLocus {
            n_alleles,
            major_allele_freqs,
            minor_allele_freqs,
            mu_allele_freqs,
            sd_allele_freqs,
        })
    }
    pub fn allele_freq_per_chromosome(
        &self,
        allele_freqs_per_locus: &AlleleFrequencyPerLocus,
        window_bp: usize,
        is_major_allele: bool,
    ) -> Result<AlleleFrequencyPerChromosome> {
        self.check()?;
        let freqs: Vec<f32> = if is_major_allele {
            allele_freqs_per_locus.major_allele_freqs.to_owned()
        } else {
            allele_freqs_per_locus.minor_allele_freqs.to_owned()
        };
        ensure!(
            self.loci.len() == freqs.len(),
            "The dataset and vector of allele frequency per locus do not match!"
        );
        let n_loci: usize = self.loci.len();
        let n_chromosomes: usize = self.genome.len();
        let idx_loci: Vec<usize> = self.sortperm_loci()?;
        ensure!(
            n_loci == idx_loci.len(),
            "The locus sorting did not generate the expected number of loci!"
        );
        let mut chrom: &str = &self.genome[self.loci[idx_loci[0]].chromosome_id].name;
        let mut window_start_pos: usize = self.loci[idx_loci[0]].position;
        let mut positions: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_chromosomes); // start and end positions of each window
        let mut afs_mu: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // means per window
        let mut afs_sd: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // standard deviations per window
        let mut afs_n: Vec<Vec<usize>> = Vec::with_capacity(n_chromosomes); // number of loci per window
        let mut per_chromosome_pos: Vec<(usize, usize)> = Vec::new();
        let mut per_chromosome_mu: Vec<f32> = Vec::new();
        let mut per_chromosome_sd: Vec<f32> = Vec::new();
        let mut per_chromosome_n: Vec<usize> = Vec::new();
        let mut per_windows: Vec<f32> = Vec::new();
        for (i, &j) in idx_loci.iter().enumerate() {
            let locus: &Locus = &self.loci[j];
            let is_same_chromosome: bool = chrom == self.genome[locus.chromosome_id].name;
            let is_within_window: bool =
                is_same_chromosome && ((locus.position - window_start_pos) <= window_bp);
            let is_last_locus: bool = i == n_loci - 1;
            let q: f32 = freqs[j];
            if is_within_window {
                per_windows.push(q);
            } else {
                let n: usize = per_windows.len();
                let mu: f32 = per_windows.iter().sum::<f32>() / (n as f32);
                let sd: f32 = (per_windows.iter().map(|&x| (x - mu).powi(2)).sum::<f32>()
                    / (n as f32))
                    .sqrt();
                // Positional info
                ensure!(
                    i > 0,
                    "Why on earth is the first index not on the same window?! This can only happen if the first chromosome or first positions are misspecified!"
                );
                let idx_previous_locus: usize = idx_loci[i - 1];
                let previous_locus_position: usize = self.loci[idx_previous_locus].position;
                per_chromosome_pos.push((window_start_pos, previous_locus_position));
                window_start_pos = locus.position;
                per_chromosome_mu.push(mu);
                per_chromosome_sd.push(sd);
                per_chromosome_n.push(n);
                per_windows = vec![q];
            }
            if !is_same_chromosome {
                chrom = &self.genome[locus.chromosome_id].name;
                positions.push(per_chromosome_pos.to_owned());
                afs_mu.push(per_chromosome_mu.to_owned());
                afs_sd.push(per_chromosome_sd.to_owned());
                afs_n.push(per_chromosome_n.to_owned());
                per_chromosome_pos = vec![];
                per_chromosome_mu = vec![];
                per_chromosome_sd = vec![];
                per_chromosome_n = vec![];
            }
            if is_last_locus {
                let n: usize = per_windows.len();
                let mu: f32 = per_windows.iter().sum::<f32>() / (n as f32);
                let sd: f32 = (per_windows.iter().map(|&x| (x - mu).powi(2)).sum::<f32>()
                    / (n as f32))
                    .sqrt();
                per_chromosome_pos.push((window_start_pos, locus.position));
                per_chromosome_mu.push(mu);
                per_chromosome_sd.push(sd);
                per_chromosome_n.push(n);
                positions.push(per_chromosome_pos.to_owned());
                afs_mu.push(per_chromosome_mu.to_owned());
                afs_sd.push(per_chromosome_sd.to_owned());
                afs_n.push(per_chromosome_n.to_owned());
            }
        }
        Ok(AlleleFrequencyPerChromosome {
            genome: self.genome.to_owned(),
            window_bp,
            positions,
            n_loci: afs_n,
            mu: afs_mu,
            sd: afs_sd,
        })
    }
    pub fn allele_freq(
        &self,
        ctx: &GpuContext,
        window_bp: usize,
    ) -> Result<(AlleleFrequencyPerChromosome, AlleleFrequencyPerChromosome)> {
        self.check()?;
        ensure!(
            window_bp > 0,
            "The window size need to be greater than zero!"
        );
        // Extract allele frequencies across loci-alleles
        let af_per_locus_per_allele: AlleleFrequencyPerLocusPerAllele =
            self.allele_freq_per_locus_allele(ctx)?;
        // Extract major and minor allele frequencies
        let af_per_locus: AlleleFrequencyPerLocus =
            self.allele_freq_per_locus(&af_per_locus_per_allele)?;
        // Extract allele frequencies per window (per chromosome)
        let af_major_per_chrom: AlleleFrequencyPerChromosome =
            self.allele_freq_per_chromosome(&af_per_locus, window_bp, true)?;
        let af_minor_per_chrom: AlleleFrequencyPerChromosome =
            self.allele_freq_per_chromosome(&af_per_locus, window_bp, false)?;
        Ok((af_major_per_chrom, af_minor_per_chrom))
    }
    pub fn diversity_per_locus(
        &self,
        allele_freqs: &AlleleFrequencyPerLocusPerAllele,
    ) -> Result<GeneticDiversityPerLocus> {
        self.check()?;
        let n_loci_alleles: usize = self.genotype_data.shape[1] as usize;
        ensure!(
            allele_freqs.homozygosity_ones.len() == n_loci_alleles,
            "The dataset and input homozygosities do not match!"
        );
        ensure!(
            allele_freqs.homozygosity_ones.len() == allele_freqs.homozygosity_zeroes.len(),
            "The homozygosities per allele per chromosome in terms of fixations as zeroes and ones do not match!"
        );
        let n_loci: usize = self.loci.len();
        let mut n_alleles: Vec<usize> = Vec::with_capacity(n_loci); // allele counts per locus
        let mut homozygosity_ones_sum: Vec<f32> = Vec::with_capacity(n_loci); // sum of frequency of fixed alleles at 1.00 across all alleles per locus (note that heterozygosity = 1.00 - homozygosity_ones_sum)
        let mut homozygosity_ones_max: Vec<f32> = Vec::with_capacity(n_loci); // max frequency of fixed allele at 1.00 across all alleles per locus
        let mut heterozygosity: Vec<f32> = Vec::with_capacity(n_loci); // heterozygosity = 1.00 - homozygosity_ones_sum
        for locus in &self.loci {
            let mut fs: Vec<f32> = Vec::with_capacity(locus.col_idx.len());
            for &j in &locus.col_idx {
                let f: f32 = allele_freqs.homozygosity_ones[j];
                ensure!(
                    (0.0..=1.0).contains(&f),
                    "Allele dosages are not consistent with the expected ploidy of the dataset (see locus: {:?})!",
                    locus
                );
                fs.push(f);
            }
            let n: usize = fs.len();
            let h_sum: f32 = fs.iter().sum::<f32>();
            let h_max: f32 = fs.iter().fold(0.0, |m, &x| m.max(x));
            let het: f32 = 1.00 - h_sum;
            fs.sort_by(|x, y| x.total_cmp(y));
            n_alleles.push(n);
            homozygosity_ones_sum.push(h_sum);
            homozygosity_ones_max.push(h_max);
            heterozygosity.push(het);
        }
        Ok(GeneticDiversityPerLocus {
            n_alleles,
            homozygosity_ones_sum,
            homozygosity_ones_max,
            heterozygosity,
        })
    }
    pub fn diversity_per_chromosome(
        &self,
        diversity_per_locus: &GeneticDiversityPerLocus,
        window_bp: usize,
    ) -> Result<GeneticDiversityPerChromosome> {
        self.check()?;
        let n: usize = diversity_per_locus.n_alleles.len();
        ensure!(
            n == diversity_per_locus.homozygosity_ones_sum.len(),
            "The allele counts per locus does not match the sum of homozygosities (fixed at 1.00)!"
        );
        ensure!(
            n == diversity_per_locus.homozygosity_ones_max.len(),
            "The allele counts per locus does not match the max of homozygosities (fixed at 1.00)!"
        );
        ensure!(
            n == diversity_per_locus.heterozygosity.len(),
            "The allele counts per locus does not match the heterozygosities!"
        );
        for (s, m) in diversity_per_locus
            .homozygosity_ones_sum
            .iter()
            .zip(diversity_per_locus.homozygosity_ones_max.iter())
        {
            ensure!(
                s >= m,
                "The sum of homozygosities (fixed at 1.00) is expected to be greater than or equal to the maximum homozygosity! Something catastrophic happened extracting GeneticDiversityPerLocus!"
            );
        }
        let n_loci: usize = self.loci.len();
        let n_chromosomes: usize = self.genome.len();
        let idx_loci: Vec<usize> = self.sortperm_loci()?;
        ensure!(
            n_loci == idx_loci.len(),
            "The locus sorting did not generate the expected number of loci!"
        );
        let mut chrom: &str = &self.genome[self.loci[idx_loci[0]].chromosome_id].name;
        let mut window_start_pos: usize = self.loci[idx_loci[0]].position;
        let mut positions: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_chromosomes); // start and end positions of each window
        let mut heterozygosity_n: Vec<Vec<usize>> = Vec::with_capacity(n_chromosomes); // means per window
        let mut heterozygosity_mu: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // means per window
        let mut heterozygosity_sd: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // means per window
        let mut per_chromosome_pos: Vec<(usize, usize)> = Vec::new();
        let mut per_chromosome_n: Vec<usize> = Vec::new();
        let mut per_chromosome_mu: Vec<f32> = Vec::new();
        let mut per_chromosome_sd: Vec<f32> = Vec::new();
        let mut per_windows: Vec<f32> = Vec::new();
        for (i, &j) in idx_loci.iter().enumerate() {
            let locus: &Locus = &self.loci[j];
            let is_same_chromosome: bool = chrom == self.genome[locus.chromosome_id].name;
            let is_within_window: bool =
                is_same_chromosome && ((locus.position - window_start_pos) <= window_bp);
            let is_last_locus: bool = i == n_loci - 1;
            let h: f32 = diversity_per_locus.heterozygosity[j];
            if is_within_window {
                per_windows.push(h);
            } else {
                let n: usize = per_windows.len();
                let mu: f32 = per_windows.iter().sum::<f32>() / (n as f32);
                let sd: f32 = (per_windows.iter().map(|&x| (x - mu).powi(2)).sum::<f32>()
                    / (n as f32))
                    .sqrt();
                // Positional info
                ensure!(
                    i > 0,
                    "Why on earth is the first index not on the same window?! This can only happen if the first chromosome or first positions are misspecified!"
                );
                let idx_previous_locus: usize = idx_loci[i - 1];
                let previous_locus_position: usize = self.loci[idx_previous_locus].position;
                per_chromosome_pos.push((window_start_pos, previous_locus_position));
                window_start_pos = locus.position;
                per_chromosome_mu.push(mu);
                per_chromosome_sd.push(sd);
                per_chromosome_n.push(n);
                per_windows = vec![h];
            }
            if !is_same_chromosome {
                chrom = &self.genome[locus.chromosome_id].name;
                positions.push(per_chromosome_pos.to_owned());
                heterozygosity_mu.push(per_chromosome_mu.to_owned());
                heterozygosity_sd.push(per_chromosome_sd.to_owned());
                heterozygosity_n.push(per_chromosome_n.to_owned());
                per_chromosome_pos = vec![];
                per_chromosome_mu = vec![];
                per_chromosome_sd = vec![];
                per_chromosome_n = vec![];
            }
            if is_last_locus {
                let n: usize = per_windows.len();
                let mu: f32 = per_windows.iter().sum::<f32>() / (n as f32);
                let sd: f32 = (per_windows.iter().map(|&x| (x - mu).powi(2)).sum::<f32>()
                    / (n as f32))
                    .sqrt();
                per_chromosome_pos.push((window_start_pos, locus.position));
                per_chromosome_mu.push(mu);
                per_chromosome_sd.push(sd);
                per_chromosome_n.push(n);
                positions.push(per_chromosome_pos.to_owned());
                heterozygosity_mu.push(per_chromosome_mu.to_owned());
                heterozygosity_sd.push(per_chromosome_sd.to_owned());
                heterozygosity_n.push(per_chromosome_n.to_owned());
            }
        }
        Ok(GeneticDiversityPerChromosome {
            genome: self.genome.to_owned(),
            window_bp,
            positions,
            n_loci: heterozygosity_n,
            heterozygosity_mu,
            heterozygosity_sd,
        })
    }
    pub fn diversity_per_entry(&self, ctx: &GpuContext) -> Result<GeneticDiversityPerEntry> {
        self.check()?;
        let n_entries: usize = self.entries.len();
        let n_loci: usize = self.loci.len();
        let n_loci_alleles: usize = self.genotype_data.shape[1] as usize;
        let kernel: GpuKernel = GpuKernel { ctx };
        let allele_dosages: GpuTensor = kernel.add(
            &self
                .genotype_data
                .slice_view(&[(0, n_entries), (0, n_loci_alleles), (0, 1)])?,
            &self
                .genotype_data
                .slice_view(&[(0, n_entries), (0, n_loci_alleles), (1, 2)])?,
        )?; // n_entries x n_loci_alleles
        let alleles_fixed_at_one: GpuTensor =
            kernel.eq_scalar(&allele_dosages, self.ploidy as f32)?; // n_entries x n_loci_alleles
        let ones: GpuTensor = GpuTensor::from_vec_f32(
            ctx,
            &vec![1.0; n_loci_alleles],
            &[n_loci_alleles as u32, 1],
            None,
            None,
        )?;
        let homozygosity: Vec<f32> = kernel
            .div_scalar_matrix(
                &kernel.matmul(&alleles_fixed_at_one, &ones)?,
                n_loci as f32, // we only divide by n_loci because the frequency of each allele per locus sums up to 1.00, hence the maximum number of loci-alleles fixed at 1.00 can only be n_loci not n_loci_alleles
            )?
            .to_vec_f32(ctx)?; // n_entries
        let heterozygosity: Vec<f32> = homozygosity.iter().map(|&x| 1.00 - x).collect();

        // Output
        Ok(GeneticDiversityPerEntry {
            entries: self.entries.to_owned(),
            n_loci,
            homozygosity,
            heterozygosity,
        })
    }

    pub fn diversity(
        &self,
        ctx: &GpuContext,
        window_bp: usize,
    ) -> Result<(
        GeneticDiversityPerLocus,
        GeneticDiversityPerChromosome,
        GeneticDiversityPerEntry,
    )> {
        self.check()?;
        ensure!(
            window_bp > 0,
            "The window size need to be greater than zero!"
        );
        let af_per_locus_per_allele: AlleleFrequencyPerLocusPerAllele =
            self.allele_freq_per_locus_allele(ctx)?;
        let diversity_per_locus: GeneticDiversityPerLocus =
            self.diversity_per_locus(&af_per_locus_per_allele)?;
        let diversity_per_chrom: GeneticDiversityPerChromosome =
            self.diversity_per_chromosome(&diversity_per_locus, window_bp)?;
        let diversity_per_entry = self.diversity_per_entry(ctx)?;
        Ok((
            diversity_per_locus,
            diversity_per_chrom,
            diversity_per_entry,
        ))
    }
    pub fn ld_map(&self) -> Result<()> {
        todo!()
    }
    pub fn trait_dist(&self) -> Result<()> {
        todo!()
    }
    pub fn trait_corr(&self) -> Result<()> {
        todo!()
    }
    pub fn stats(&self) -> Result<DatasetStats> {
        self.check()?;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }
    fn dataset(ctx: &GpuContext) -> Dataset {
        Dataset::sim_founders(
            ctx, 500, 2, true, 5, 200, 3, &[0.5; 3], None, None, None, None, 42,
        )
        .unwrap()
        .0
    }
    #[test]
    fn sortperm_entries_returns_valid_permutation() {
        let ctx = context();
        let ds = dataset(&ctx);
        let idx = ds.sortperm_entries().unwrap();
        assert_eq!(idx.len(), ds.entries.len());
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..ds.entries.len()).collect::<Vec<_>>());
    }
    #[test]
    fn sortperm_entries_generates_sorted_entries() {
        let ctx = context();
        let ds = dataset(&ctx);
        let idx = ds.sortperm_entries().unwrap();
        for w in idx.windows(2) {
            assert!(ds.entries[w[0]] <= ds.entries[w[1]]);
        }
    }
    #[test]
    fn sortperm_loci_returns_valid_permutation() {
        let ctx = context();
        let ds = dataset(&ctx);
        let idx = ds.sortperm_loci().unwrap();
        assert_eq!(idx.len(), ds.loci.len());
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..ds.loci.len()).collect::<Vec<_>>());
    }
    #[test]
    fn sortperm_loci_generates_sorted_loci() {
        let ctx = context();
        let ds = dataset(&ctx);
        let idx = ds.sortperm_loci().unwrap();
        for w in idx.windows(2) {
            assert!(ds.loci[w[0]] <= ds.loci[w[1]]);
        }
    }
    #[test]
    fn allele_freq_per_locus_allele_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let n = ds.genotype_data.shape[1] as usize;
        assert_eq!(af.allele_freqs.len(), n);
        assert_eq!(af.homozygosity_zeroes.len(), n);
        assert_eq!(af.homozygosity_ones.len(), n);
    }
    #[test]
    fn allele_freq_per_locus_allele_values_are_bounded() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        for &x in &af.allele_freqs {
            assert!((0.0..=1.0).contains(&x));
        }
        for &x in &af.homozygosity_zeroes {
            assert!((0.0..=1.0).contains(&x));
        }
        for &x in &af.homozygosity_ones {
            assert!((0.0..=1.0).contains(&x));
        }
    }
    #[test]
    fn allele_freq_per_locus_allele_homozygosities_are_consistent() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        for i in 0..af.allele_freqs.len() {
            assert!(af.homozygosity_zeroes[i] + af.homozygosity_ones[i] <= 1.0 + 1e-6);
        }
    }
    #[test]
    fn allele_freq_per_locus_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let n_loci = ds.loci.len();
        assert_eq!(af1.n_alleles.len(), n_loci);
        assert_eq!(af1.major_allele_freqs.len(), n_loci);
        assert_eq!(af1.minor_allele_freqs.len(), n_loci);
        assert_eq!(af1.mu_allele_freqs.len(), n_loci);
        assert_eq!(af1.sd_allele_freqs.len(), n_loci);
    }
    #[test]
    fn allele_freq_per_locus_n_alleles_matches_locus_definition() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        for (i, locus) in ds.loci.iter().enumerate() {
            assert_eq!(af1.n_alleles[i], locus.col_idx.len());
        }
    }
    #[test]
    fn allele_freq_per_locus_major_is_not_less_than_minor() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        for i in 0..ds.loci.len() {
            assert!(af1.major_allele_freqs[i] >= af1.minor_allele_freqs[i]);
        }
    }
    #[test]
    fn allele_freq_per_locus_statistics_are_bounded() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        for &x in &af1.major_allele_freqs {
            assert!((0.0..=1.0).contains(&x));
        }
        for &x in &af1.minor_allele_freqs {
            assert!((0.0..=1.0).contains(&x));
        }
        for &x in &af1.mu_allele_freqs {
            assert!((0.0..=1.0).contains(&x));
        }
        for &x in &af1.sd_allele_freqs {
            assert!(x >= 0.0);
            assert!(x.is_finite());
        }
    }
    #[test]
    fn allele_freq_per_chromosome_major_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        assert_eq!(af2.positions.len(), ds.genome.len());
        assert_eq!(af2.n_loci.len(), ds.genome.len());
        assert_eq!(af2.mu.len(), ds.genome.len());
        assert_eq!(af2.sd.len(), ds.genome.len());
    }
    #[test]
    fn allele_freq_per_chromosome_minor_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, false).unwrap();
        assert_eq!(af2.positions.len(), ds.genome.len());
        assert_eq!(af2.n_loci.len(), ds.genome.len());
        assert_eq!(af2.mu.len(), ds.genome.len());
        assert_eq!(af2.sd.len(), ds.genome.len());
    }
    #[test]
    fn allele_freq_per_chromosome_window_shapes_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        for i in 0..af2.positions.len() {
            assert_eq!(af2.positions[i].len(), af2.n_loci[i].len());
            assert_eq!(af2.positions[i].len(), af2.mu[i].len());
            assert_eq!(af2.positions[i].len(), af2.sd[i].len());
        }
    }
    #[test]
    fn allele_freq_per_chromosome_window_counts_are_positive() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        for chrom in &af2.n_loci {
            for &n in chrom {
                assert!(n > 0);
            }
        }
    }
    #[test]
    fn allele_freq_per_chromosome_coordinates_are_valid() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        for chrom in &af2.positions {
            for &(start, end) in chrom {
                assert!(end >= start, "invalid interval: ({start}, {end})");
            }
        }
    }
    #[test]
    fn allele_freq_per_chromosome_coordinates_are_monotonic() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        for chrom in &af2.positions {
            for w in chrom.windows(2) {
                assert!(w[1].0 >= w[0].0);
                assert!(w[1].1 >= w[0].1);
            }
        }
    }
    #[test]
    fn allele_freq_per_chromosome_statistics_are_finite() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af0 = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let af1 = ds.allele_freq_per_locus(&af0).unwrap();
        let af2 = ds.allele_freq_per_chromosome(&af1, 100_000, true).unwrap();
        for chrom in &af2.mu {
            for &x in chrom {
                assert!(x.is_finite());
            }
        }
        for chrom in &af2.sd {
            for &x in chrom {
                assert!(x.is_finite());
                assert!(x >= 0.0);
            }
        }
    }
    #[test]
    fn allele_freq_runs() {
        let ctx = context();
        let ds = dataset(&ctx);
        let (major, minor) = ds.allele_freq(&ctx, 100_000).unwrap();
        assert_eq!(major.genome.len(), ds.genome.len());
        assert_eq!(minor.genome.len(), ds.genome.len());
    }
    #[test]
    fn allele_freq_major_minor_shapes_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let (major, minor) = ds.allele_freq(&ctx, 100_000).unwrap();
        assert_eq!(major.positions.len(), minor.positions.len());
        assert_eq!(major.n_loci.len(), minor.n_loci.len());
        assert_eq!(major.mu.len(), minor.mu.len());
        assert_eq!(major.sd.len(), minor.sd.len());
    }
    #[test]
    fn allele_freq_major_windows_match_minor_windows() {
        let ctx = context();
        let ds = dataset(&ctx);
        let (major, minor) = ds.allele_freq(&ctx, 100_000).unwrap();
        for i in 0..major.positions.len() {
            assert_eq!(major.positions[i].len(), minor.positions[i].len());
        }
    }
    #[test]
    fn allele_freq_rejects_zero_window_size() {
        let ctx = context();
        let ds = dataset(&ctx);
        assert!(ds.allele_freq(&ctx, 0).is_err());
    }

    #[test]
    fn diversity_per_locus_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let diversity = ds.diversity_per_locus(&af).unwrap();
        let n_loci = ds.loci.len();

        assert_eq!(diversity.n_alleles.len(), n_loci);
        assert_eq!(diversity.homozygosity_ones_sum.len(), n_loci);
        assert_eq!(diversity.homozygosity_ones_max.len(), n_loci);
        assert_eq!(diversity.heterozygosity.len(), n_loci);
    }

    #[test]
    fn diversity_per_locus_allele_counts_match_definitions() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let diversity = ds.diversity_per_locus(&af).unwrap();

        for (i, locus) in ds.loci.iter().enumerate() {
            assert_eq!(diversity.n_alleles[i], locus.col_idx.len());
        }
    }

    #[test]
    fn diversity_per_locus_heterozygosity_is_complement_of_homozygosity_sum() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let diversity = ds.diversity_per_locus(&af).unwrap();

        for i in 0..ds.loci.len() {
            let expected_het = 1.0 - diversity.homozygosity_ones_sum[i];
            assert!((diversity.heterozygosity[i] - expected_het).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn diversity_per_locus_sum_is_greater_than_or_equal_to_max() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let diversity = ds.diversity_per_locus(&af).unwrap();

        for i in 0..ds.loci.len() {
            assert!(diversity.homozygosity_ones_sum[i] >= diversity.homozygosity_ones_max[i]);
        }
    }

    #[test]
    fn diversity_per_chromosome_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let div_locus = ds.diversity_per_locus(&af).unwrap();
        let div_chrom = ds.diversity_per_chromosome(&div_locus, 100_000).unwrap();

        assert_eq!(div_chrom.positions.len(), ds.genome.len());
        assert_eq!(div_chrom.n_loci.len(), ds.genome.len());
        assert_eq!(div_chrom.heterozygosity_mu.len(), ds.genome.len());
        assert_eq!(div_chrom.heterozygosity_sd.len(), ds.genome.len());
    }

    #[test]
    fn diversity_per_chromosome_window_shapes_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let div_locus = ds.diversity_per_locus(&af).unwrap();
        let div_chrom = ds.diversity_per_chromosome(&div_locus, 100_000).unwrap();

        for i in 0..div_chrom.positions.len() {
            assert_eq!(div_chrom.positions[i].len(), div_chrom.n_loci[i].len());
            assert_eq!(
                div_chrom.positions[i].len(),
                div_chrom.heterozygosity_mu[i].len()
            );
            assert_eq!(
                div_chrom.positions[i].len(),
                div_chrom.heterozygosity_sd[i].len()
            );
        }
    }

    #[test]
    fn diversity_per_chromosome_window_counts_are_positive() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let div_locus = ds.diversity_per_locus(&af).unwrap();
        let div_chrom = ds.diversity_per_chromosome(&div_locus, 100_000).unwrap();

        for chrom in &div_chrom.n_loci {
            for &n in chrom {
                assert!(n > 0);
            }
        }
    }

    #[test]
    fn diversity_per_chromosome_coordinates_are_valid_and_monotonic() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let div_locus = ds.diversity_per_locus(&af).unwrap();
        let div_chrom = ds.diversity_per_chromosome(&div_locus, 100_000).unwrap();

        for chrom in &div_chrom.positions {
            for w in chrom.windows(2) {
                assert!(
                    w[0].1 >= w[0].0,
                    "invalid interval: ({}, {})",
                    w[0].0,
                    w[0].1
                );
                assert!(w[1].0 >= w[0].0, "non-monotonic start coordinates");
                assert!(w[1].1 >= w[0].1, "non-monotonic end coordinates");
            }
        }
    }

    #[test]
    fn diversity_per_chromosome_statistics_are_finite() {
        let ctx = context();
        let ds = dataset(&ctx);
        let af = ds.allele_freq_per_locus_allele(&ctx).unwrap();
        let div_locus = ds.diversity_per_locus(&af).unwrap();
        let div_chrom = ds.diversity_per_chromosome(&div_locus, 100_000).unwrap();

        for chrom in &div_chrom.heterozygosity_mu {
            for &x in chrom {
                assert!(x.is_finite());
            }
        }
        for chrom in &div_chrom.heterozygosity_sd {
            for &x in chrom {
                assert!(x.is_finite());
                assert!(x >= 0.0);
            }
        }
    }

    #[test]
    fn diversity_per_entry_dimensions_match() {
        let ctx = context();
        let ds = dataset(&ctx);
        let div_entry = ds.diversity_per_entry(&ctx).unwrap();
        let n_entries = ds.entries.len();

        assert_eq!(div_entry.entries.len(), n_entries);
        assert_eq!(div_entry.homozygosity.len(), n_entries);
        assert_eq!(div_entry.heterozygosity.len(), n_entries);
        assert_eq!(div_entry.n_loci, ds.loci.len());
    }

    #[test]
    fn diversity_per_entry_heterozygosity_is_complement_of_homozygosity() {
        let ctx = context();
        let ds = dataset(&ctx);
        let div_entry = ds.diversity_per_entry(&ctx).unwrap();

        for i in 0..ds.entries.len() {
            let expected_het = 1.0 - div_entry.homozygosity[i];
            assert!((div_entry.heterozygosity[i] - expected_het).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn diversity_per_entry_statistics_are_bounded() {
        let ctx = context();
        let ds = dataset(&ctx);
        let div_entry = ds.diversity_per_entry(&ctx).unwrap();

        for &val in &div_entry.homozygosity {
            assert!((0.0..=1.0).contains(&val));
        }
        for &val in &div_entry.heterozygosity {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn diversity_pipeline_runs_successfully() {
        let ctx = context();
        let ds = dataset(&ctx);
        let (div_locus, div_chrom, div_entry) = ds.diversity(&ctx, 100_000).unwrap();

        assert_eq!(div_locus.heterozygosity.len(), ds.loci.len());
        assert_eq!(div_chrom.positions.len(), ds.genome.len());
        assert_eq!(div_entry.heterozygosity.len(), ds.entries.len());
    }

    #[test]
    fn diversity_pipeline_rejects_zero_window_size() {
        let ctx = context();
        let ds = dataset(&ctx);
        assert!(ds.diversity(&ctx, 0).is_err());
    }
}
