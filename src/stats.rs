use anyhow::{Result, ensure};
use std::fmt;
use crate::{
    dataset::Dataset,
    linalg::{context::GpuContext, kernel::GpuKernel, tensor::GpuTensor},
};

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
    pub fn allele_freq_spec(&self, ctx: &GpuContext, window_bp: usize) -> Result<()> {
        self.check()?;
        ensure!(window_bp > 0, "The window size need to be greater than zero!");
        let idx_entries: Vec<usize> = self.sortperm_entries()?;
        // let idx_loci: Vec<usize> = self.sortperm_loci()?;
        // let n_chromosomes: usize = self.genome.len();
        let n_entries: usize = idx_entries.len();
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
            .div_scalar_matrix(&kernel.matmul(&allele_dosages, &ones)?, n_entries as f32)?
            .to_vec_f32(ctx)?; //n_loci_alleles
        println!("allele_freqs: {:?}", allele_freqs);
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
