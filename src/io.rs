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
            header.push(t.name.as_str());
        }
        writeln!(file_buffer, "{}", header.join(delimiter))?;
        // Phenotype values
        let phenotype_vec = self.phenotype_data.to_vec_f32(ctx)?; // maybe a large allocation if phenotype data is large
        for i in 0..n_entries {
            let mut line: Vec<String> = Vec::with_capacity(3 + n_traits);
            line.push(self.entries[i].name.to_owned());
            line.push(self.entries[i].species.to_owned());
            line.push(self.entries[i].group.to_owned());
            for j in 0..n_traits {
                line.push(phenotype_vec[(i * n_traits) + j].to_string());
            }
            writeln!(file_buffer, "{}", line.join(delimiter))?;
        }
        file_buffer.flush()?; // Explicitly flush to make sure we successfully wrote into disk!
        Ok(())
    }

    // pub fn write_gneotype(
    //     &self,
    //     ctx: &GpuContext,
    //     fname: &str,
    //     delimiter: &str,
    //     overwrite: bool,
    // ) -> Result<()> {
    //     todo!()
    // }
}
