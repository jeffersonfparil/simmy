use crate::genotype::Locus;
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

pub struct AlleleFrequencySpectrum {
    major_allele_freqs: Vec<f32>, // max
    minor_allele_freqs: Vec<f32>, // min
    mu_allele_freqs: Vec<f32>,    // mean
    sd_allele_freqs: Vec<f32>,    // sd
    md_allele_freqs: Vec<f32>,    // median
    window_bp: usize,
}

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
    pub fn allele_freq_per_locus_allele(&self, ctx: &GpuContext) -> Result<Vec<f32>> {
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
        Ok(allele_freqs)
    }
    pub fn allele_freq_major_minor_etc(
        &self,
        allele_freqs: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.check()?;
        let n_loci_alleles: usize = self.genotype_data.shape[1] as usize;
        ensure!(
            allele_freqs.len() == n_loci_alleles,
            "The dataset and input allele frequencies do not match!"
        );
        let n_loci: usize = self.loci.len();
        let mut major_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci);
        let mut minor_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci);
        let mut mu_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci);
        let mut sd_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci);
        let mut md_allele_freqs: Vec<f32> = Vec::with_capacity(n_loci);
        for locus in &self.loci {
            let mut p: f32 = 0.0;
            let mut q: f32 = 1.0;
            let mut fs: Vec<f32> = Vec::with_capacity(locus.col_idx.len());
            for &j in &locus.col_idx {
                let f: f32 = allele_freqs[j];
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
            fs.sort_by(|x, y| x.total_cmp(y));
            let md: f32 = fs[n / 2];
            major_allele_freqs.push(p);
            minor_allele_freqs.push(q);
            mu_allele_freqs.push(mu);
            sd_allele_freqs.push(sd);
            md_allele_freqs.push(md);
        }
        Ok((
            major_allele_freqs,
            minor_allele_freqs,
            mu_allele_freqs,
            sd_allele_freqs,
            md_allele_freqs,
        ))
    }
    pub fn allele_freq_summary(
        &self,
        allele_freqs_across_loci: &[f32],
        window_bp: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>)> {
        self.check()?;
        ensure!(
            self.loci.len() == allele_freqs_across_loci.len(),
            "The dataset and vector of allele frequency per locus do not match!"
        );
        let n_loci: usize = self.loci.len();
        let n_chromosomes: usize = self.genome.len();
        let idx_loci: Vec<usize> = self.sortperm_loci()?;
        let mut chrom: &str = &self.genome[self.loci[idx_loci[0]].chromosome_id].name;
        let mut window_start_pos: usize = self.loci[idx_loci[0]].position;
        let mut afs_mu: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // means per window
        let mut afs_sd: Vec<Vec<f32>> = Vec::with_capacity(n_chromosomes); // standard deviations per window
        let mut afs_n: Vec<Vec<usize>> = Vec::with_capacity(n_chromosomes); // number of loci per window
        let mut freqs_per_chromosome_mu: Vec<f32> = Vec::new();
        let mut freqs_per_chromosome_sd: Vec<f32> = Vec::new();
        let mut freqs_per_chromosome_n: Vec<usize> = Vec::new();
        let mut freqs_per_windows: Vec<f32> = Vec::new();
        for &j in &idx_loci {
            let locus: &Locus = &self.loci[j];
            let is_same_chromosome: bool = chrom == &self.genome[locus.chromosome_id].name;
            let is_within_window: bool = (locus.position - window_start_pos) <= window_bp;
            let is_last_locus: bool = j == idx_loci[n_loci - 1];


            // Continue refactoring here...


            if chrom == &self.genome[locus.chromosome_id].name {
                if ((locus.position - window_start_pos) <= window_bp) && !is_last_locus {
                    freqs_per_windows.push(allele_freqs_across_loci[j]);
                } else {
                    window_start_pos = locus.position;
                    let n: usize = freqs_per_windows.len();
                    let mu: f32 = freqs_per_windows.iter().sum::<f32>() / (n as f32);
                    let sd: f32 = freqs_per_windows
                        .iter()
                        .map(|&x| (x - mu).powi(2))
                        .sum::<f32>()
                        / (n as f32);
                    freqs_per_chromosome_mu.push(mu);
                    freqs_per_chromosome_sd.push(sd);
                    freqs_per_chromosome_n.push(n);
                    freqs_per_windows = vec![allele_freqs_across_loci[j]];
                }
            } else {
                chrom = &self.genome[locus.chromosome_id].name;
                afs_mu.push(freqs_per_chromosome_mu.to_owned());
                afs_sd.push(freqs_per_chromosome_sd.to_owned());
                afs_n.push(freqs_per_chromosome_n.to_owned());
                freqs_per_chromosome_mu = vec![];
                freqs_per_chromosome_sd = vec![];
                freqs_per_chromosome_n = vec![];
            }
        }
        Ok((afs_mu, afs_sd, afs_n))
    }
    pub fn allele_freq_spec(&self, ctx: &GpuContext, window_bp: usize) -> Result<()> {
        self.check()?;
        ensure!(
            window_bp > 0,
            "The window size need to be greater than zero!"
        );
        // Extract allele frequencies across loci-alleles
        let allele_freqs: Vec<f32> = self.allele_freq_per_locus_allele(ctx)?;
        // Extract major and minor allele frequencies
        let (
            major_allele_freqs,
            minor_allele_freqs,
            mu_allele_freqs,
            sd_allele_freqs,
            md_allele_freqs,
        ) = self.allele_freq_major_minor_etc(&allele_freqs)?;
        // Extract allele frequencies per window (per chromosome)
        self.allele_freq_summary(&major_allele_freqs, window_bp)?;
        self.allele_freq_summary(&minor_allele_freqs, window_bp)?;
        self.allele_freq_summary(&mu_allele_freqs, window_bp)?;
        self.allele_freq_summary(&sd_allele_freqs, window_bp)?;
        self.allele_freq_summary(&md_allele_freqs, window_bp)?;

        todo!()
    }
    pub fn ld_map(&self) -> Result<()> {
        todo!()
    }
    pub fn inbreeding(&self) -> Result<()> {
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
