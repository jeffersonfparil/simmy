use crate::dataset::Dataset;
use crate::linalg::context::GpuContext;
use anyhow::{Result, bail};
use std::fs::{File, OpenOptions, exists, remove_file};
use std::io::{BufWriter, Write};

pub fn open_file(fname: &str, overwrite: bool) -> Result<BufWriter<File>> {
    if exists(fname)? {
        if overwrite {
            remove_file(fname)?;
        } else {
            bail!("The file:\"{}\" exists but overwrite = false!", fname);
        }
    }
    let file: File = OpenOptions::new()
        .create_new(true) // Errors if file exists!
        .write(true)
        .open(fname)?;
    // Using buffer for large phenotype tables
    Ok(BufWriter::new(file))
}

pub fn check_strings(string: &str, delimiter: &str) -> Result<()> {
    let illegal_strings: Vec<&str> = vec!["\"", "\n", "|", delimiter];
    for ill in illegal_strings {
        if string.contains(ill) {
            bail!("The string: {} contains illegal string: {:?}", string, ill);
        }
    }
    Ok(())
}

impl Dataset {
    pub fn write_phenotype(
        &self,
        ctx: &GpuContext,
        fname: &str,
        delimiter: &str,
        overwrite: bool,
    ) -> Result<()> {
        // Generates a complete phenotype table file
        // Not intended to append to an existing phenotype table file
        self.check_dimensions()?;
        let n_entries: usize = self.entries.len();
        let n_traits: usize = self.traits.len();
        // Open file
        let mut file_buffer: BufWriter<File> = open_file(fname, overwrite)?;
        // Header
        let mut header: Vec<&str> = Vec::with_capacity(3 + n_traits);
        header.push("name");
        header.push("species");
        header.push("group");
        for t in self.traits.iter() {
            check_strings(&t.name, delimiter)?;
            header.push(t.name.as_str());
        }
        writeln!(file_buffer, "{}", header.join(delimiter))?;
        // Phenotype values
        let phenotype_vec: Vec<f32> = self.phenotype_data.to_vec_f32(ctx)?; // maybe a large allocation if phenotype data is large
        let mut line: Vec<String> = vec!["".to_owned(); 3 + n_traits];
        for i in 0..n_entries {
            check_strings(&self.entries[i].name, delimiter)?;
            line[0] = self.entries[i].name.to_owned();
            check_strings(&self.entries[i].species, delimiter)?;
            line[1] = self.entries[i].species.to_owned();
            check_strings(&self.entries[i].group, delimiter)?;
            line[2] = self.entries[i].group.to_owned();
            for j in 0..n_traits {
                line[3 + j] = phenotype_vec[(i * n_traits) + j].to_string();
            }
            writeln!(file_buffer, "{}", line.join(delimiter))?;
        }
        file_buffer.flush()?; // Explicitly flush to make sure we successfully wrote into disk!
        Ok(())
    }
    pub fn read_phenotype(&mut self) -> Result<()> {
        todo!()
    }
    pub fn write_genotype(
        &self,
        ctx: &GpuContext,
        fname: &str,
        delimiter: &str,
        overwrite: bool,
    ) -> Result<()> {
        // Generates a complete genotype table file
        // Not intended to append to an existing genotype table file
        self.check_dimensions()?;
        let n_entries: usize = self.entries.len();
        let n_loci_alleles: usize = self.loci.iter().map(|l| l.col_idx.len()).sum();
        // Open file
        let mut file_buffer: BufWriter<File> = open_file(fname, overwrite)?;
        // Check chromosome names
        for chrom in self.genome.iter() {
            check_strings(&chrom.name, delimiter)?;
        }
        // Header
        let mut header: Vec<String> = vec!["".to_owned(); 3 + n_loci_alleles];
        header[0] = "name".to_owned();
        header[1] = "species".to_owned();
        header[2] = "group".to_owned();
        for locus in self.loci.iter() {
            for (&j, allele) in locus.col_idx.iter().zip(&locus.alleles) {
                check_strings(allele, delimiter)?;
                header[3 + j] = format!(
                    "{}|{}|{}",
                    self.genome[locus.chromosome_id].name, locus.position, allele
                );
            }
        }
        writeln!(file_buffer, "{}", header.join(delimiter))?;
        // Genotype values
        let genotype_vec: Vec<f32> = self.genotype_data.to_vec_f32(ctx)?; // maybe a large allocation if genotype data is large
        let mut line: Vec<String> = vec!["".to_owned(); 3 + n_loci_alleles];
        for i in 0..n_entries {
            check_strings(&self.entries[i].name, delimiter)?;
            line[0] = self.entries[i].name.to_owned();
            check_strings(&self.entries[i].species, delimiter)?;
            line[1] = self.entries[i].species.to_owned();
            check_strings(&self.entries[i].group, delimiter)?;
            line[2] = self.entries[i].group.to_owned();
            for locus in self.loci.iter() {
                for &j in locus.col_idx.iter() {
                    let idx_src: usize = (i * n_loci_alleles * 2) + (j * 2);
                    let idx_des: usize = 3 + j;
                    let a_0: String = genotype_vec[idx_src].to_string();
                    let a_1: String = genotype_vec[idx_src + 1].to_string();
                    line[idx_des] = a_0 + "/" + &a_1;
                }
            }
            writeln!(file_buffer, "{}", line.join(delimiter))?;
        }
        file_buffer.flush()?; // Explicitly flush to make sure we successfully wrote into disk!
        Ok(())
    }
    pub fn read_genotype(&mut self) -> Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::linalg::context::GpuContext;
    use std::fs::{read_to_string, remove_file};
    use std::path::PathBuf;

    fn context() -> GpuContext {
        pollster::block_on(GpuContext::new()).expect("Failed to create GPU context")
    }

    fn test_dataset(ctx: &GpuContext) -> Dataset {
        let (parents, _mapper) = Dataset::sim_founders(ctx, 1000, 5, 200, 5, 2, true, 42).unwrap();
        parents
    }

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn rejects_delimiter() {
        assert!(check_strings("abc\tdef", "\t").is_err());
    }

    #[test]
    fn write_phenotype_creates_file() {
        let ctx = context();
        let ds = test_dataset(&ctx);
        let path = temp_file("phenotype_create.tsv");
        ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();
        assert!(path.exists());
        let _ = remove_file(&path);
    }

    #[test]
    fn write_phenotype_writes_all_entries() {
        let ctx = context();
        let ds = test_dataset(&ctx);
        let path = temp_file("phenotype_rows.tsv");
        ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();
        let text = read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), ds.entries.len() + 1);
        let _ = remove_file(&path);
    }

    #[test]
    fn genotype_header_has_expected_number_of_columns() {
        let ctx = context();
        let ds = test_dataset(&ctx);
        let path = temp_file("genotype_header.tsv");
        ds.write_genotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();
        let text = read_to_string(&path).unwrap();
        let header = text.lines().next().unwrap();
        let cols: Vec<_> = header.split('\t').collect();
        let expected = 3 + ds.loci.iter().map(|l| l.col_idx.len()).sum::<usize>();
        assert_eq!(cols.len(), expected);
        let _ = remove_file(&path);
    }

    #[test]
    fn genotype_cells_contain_single_haplotype_separator() {
        let ctx = context();
        let ds = test_dataset(&ctx);
        let path = temp_file("genotype_separator.tsv");
        ds.write_genotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();
        let text = read_to_string(&path).unwrap();
        let row = text
            .lines()
            .nth(1)
            .expect("Expected at least one genotype row");
        for value in row.split('\t').skip(3) {
            assert_eq!(
                value.matches('/').count(),
                1,
                "Malformed genotype field: {}",
                value
            );
        }
        let _ = remove_file(&path);
    }

    #[test]
    fn exported_genotypes_match_tensor_values() {
        let ctx = context();
        let ds = test_dataset(&ctx);
        let genotype = ds.genotype_data.to_vec_f32(&ctx).unwrap();
        let path = temp_file("genotype_values.tsv");
        ds.write_genotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();
        let text = read_to_string(&path).unwrap();
        let row = text.lines().nth(1).unwrap();
        let cols: Vec<_> = row.split('\t').collect();
        let expected = format!("{}/{}", genotype[0], genotype[1]);
        assert_eq!(cols[3], expected);
        let _ = remove_file(&path);
    }
}
