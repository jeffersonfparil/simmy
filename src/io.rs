use crate::dataset::Dataset;
use crate::entry::Entry;
use crate::genotype::Locus;
use crate::linalg::context::GpuContext;
use crate::linalg::tensor::GpuTensor;
use crate::phenotype::Trait;
use anyhow::{Result, bail, ensure};
use std::fs::{File, OpenOptions, exists, remove_file};
use std::io::{BufRead, BufReader, BufWriter, Write};

pub fn open_file_writer(fname: &str, overwrite: bool) -> Result<BufWriter<File>> {
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
    Ok(BufWriter::new(file))
}

pub fn open_file_reader(fname: &str) -> Result<BufReader<File>> {
    if !exists(fname)? {
        bail!("The file:\"{}\" does not exist!", fname);
    }
    let file: File = File::open(fname)?;
    Ok(BufReader::new(file))
}

pub fn check_strings(string: &str, delimiter: &str) -> Result<()> {
    for ill in ["\"", "\n", "|", delimiter] {
        if string.contains(ill) {
            bail!("The string: {} contains illegal string: {:?}", string, ill);
        }
    }
    Ok(())
}

pub fn check_header_id_cols(line: &[String], fname: &str) -> Result<()> {
    ensure!(
        line[0] == "name",
        "The first column of the file: \"{}\" should be \"name\"!",
        fname
    );
    ensure!(
        line[1] == "species",
        "The second column of the file: \"{}\" should be \"species\"!",
        fname
    );
    ensure!(
        line[2] == "group",
        "The third column of the file: \"{}\" should be \"group\"!",
        fname
    );
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
        let mut file: BufWriter<File> = open_file_writer(fname, overwrite)?;
        // Header
        let mut header: Vec<&str> = Vec::with_capacity(3 + n_traits);
        header.push("name");
        header.push("species");
        header.push("group");
        for t in self.traits.iter() {
            check_strings(&t.name, delimiter)?;
            header.push(t.name.as_str());
        }
        writeln!(file, "{}", header.join(delimiter))?;
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
            writeln!(file, "{}", line.join(delimiter))?;
        }
        file.flush()?; // Explicitly flush to make sure we successfully wrote into disk!
        Ok(())
    }
    pub fn read_phenotype(
        &mut self,
        ctx: &GpuContext,
        fname: &str,
        delimiter: &str,
        entries_exist: bool,
    ) -> Result<()> {
        let file: BufReader<File> = open_file_reader(fname)?;
        if !entries_exist {
            self.entries = Vec::new();
        }
        let mut traits: Vec<Trait> = Vec::new();
        let mut phenotype_vec: Vec<f32> = Vec::new();
        for (i, line) in file.lines().enumerate() {
            let line: Vec<String> = line?.split(delimiter).map(|x| x.to_owned()).collect();
            ensure!(
                line.len() >= 4,
                "Malformed file: \"{}\" at line {}!",
                fname,
                i
            );
            if i == 0 {
                // Header
                check_header_id_cols(&line, fname)?;
                for x in line.iter().skip(3) {
                    let t = Trait {
                        name: x.to_owned(),
                        description: "".to_owned(),
                    };
                    traits.push(t);
                }
                ensure!(
                    !traits.is_empty(),
                    "There are no traits in file: \"{}\"!",
                    fname
                );
            } else {
                // Data after the header line
                if entries_exist {
                    ensure!(
                        (i - 1) < self.entries.len(),
                        "There are too many entries in file: \"{}\" than expected!",
                        fname
                    ); // Note that we use i-1 because ith line is the header in the file
                    ensure!(
                        self.entries[i - 1].name == line[0],
                        "Entries in file: \"{}\" do not match existing Dataset! See line {}.",
                        fname,
                        i
                    ); // Again note that we use i-1 because ith line is the header in the file
                } else {
                    let entry = Entry {
                        name: line[0].to_owned(),
                        species: line[1].to_owned(),
                        group: line[2].to_owned(),
                        notes: "".to_owned(),
                    };
                    self.entries.push(entry);
                }
                ensure!(
                    line.len() == 3 + traits.len(),
                    "Malformed file: \"{}\" at line {}!",
                    fname,
                    i
                );
                for x in line.iter().skip(3) {
                    let y: f32 = x.parse::<f32>()?;
                    phenotype_vec.push(y);
                }
            }
        }
        let n_traits: u32 = traits.len() as u32;
        let n_entries: u32 = phenotype_vec.len() as u32 / n_traits;
        ensure!(
            n_entries == self.entries.len() as u32,
            "There are too few entries in file: \"{}\" than expected!",
            fname
        );
        self.traits = traits;
        self.phenotype_data =
            GpuTensor::from_vec_f32(ctx, &phenotype_vec, &[n_entries, n_traits], None, None)?;
        // Note that the resulting mutated Dataset struct may be:
        // ✅ entries
        // ❌ ploidy
        // ❌ sexes
        // ❌ genome
        // ❌ loci
        // ❌ haplotype_persistence_probs
        // ✅ traits
        // ❌ genotype_data
        // ✅ phenotype_data
        Ok(())
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
        let mut file: BufWriter<File> = open_file_writer(fname, overwrite)?;
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
        writeln!(file, "{}", header.join(delimiter))?;
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
                    line[idx_des] = a_0 + "|" + &a_1;
                }
            }
            writeln!(file, "{}", line.join(delimiter))?;
        }
        file.flush()?; // Explicitly flush to make sure we successfully wrote into disk!
        Ok(())
    }
    pub fn read_genotype(
        &mut self,
        ctx: &GpuContext,
        fname: &str,
        delimiter: &str,
        entries_exist: bool,
    ) -> Result<()> {
        ensure!(
            !self.genome.is_empty(),
            "We expect the genome information to already exist as the genotype table file does not contain all the necessary information, e.g. chromosome lengths!"
        );
        let file: BufReader<File> = open_file_reader(fname)?;
        if !entries_exist {
            self.entries = Vec::new();
        }
        let mut genotype_vec: Vec<f32> = Vec::new();
        for (i, line) in file.lines().enumerate() {
            let line: Vec<String> = line?.split(delimiter).map(|x| x.to_owned()).collect();
            ensure!(
                line.len() >= 4,
                "Malformed file: \"{}\" at line {}!",
                fname,
                i
            );
            if i == 0 {
                // Header line: parse and validate locus-allele information
                check_header_id_cols(&line, fname)?; // First 3 columns are: name, species, and group
                let mut loci: Vec<Locus> = Vec::new();
                let mut locus = Locus {
                    chromosome_id: 0,
                    position: 0,
                    alleles: vec![],
                    length: 1,
                    col_idx: vec![],
                };
                for (j, x) in line.iter().skip(3).enumerate() {
                    let y: Vec<&str> = x.split("|").collect();
                    ensure!(
                        y.len() == 3,
                        "Malformed header in file: \"{}\" at column {}. The locus-allele names should have the form: \"chromosome_name|position|allele\", e.g. \"chr_1|123|A\"!",
                        fname,
                        j + 3
                    );
                    let chromosome_name: &str = y[0];
                    let position: usize = y[1].parse::<usize>()?;
                    let allele: &str = y[2];
                    let mut okay: bool = false;
                    let mut chromosome_id: usize = 0;
                    for (k, g) in self.genome.iter().enumerate() {
                        if (g.name == chromosome_name) && (g.length > position) {
                            okay = true;
                            chromosome_id = k;
                            break;
                        }
                    }
                    ensure!(
                        okay,
                        "The chromosome name and/or length in file: \"{}\" at column {} do not match the expected information!",
                        fname,
                        j + 3
                    );
                    if !((locus.chromosome_id == chromosome_id) && (locus.position == position)) {
                        // New locus
                        if j > 0 {
                            // Push previous fully defined locus
                            loci.push(locus.clone());
                        }
                        locus.chromosome_id = chromosome_id;
                        locus.position = position;
                        locus.alleles = vec![allele.to_owned()];
                        locus.length = allele.len();
                        locus.col_idx = vec![j];
                    } else {
                        // Additional alleles at a locus
                        locus.alleles.push(allele.to_owned());
                        locus.length = [locus.length, allele.len()].into_iter().max().unwrap_or(1);
                        locus.col_idx.push(j);
                    }
                }
                // Push the last fully defined locus
                ensure!(
                    !loci.is_empty() && !locus.alleles.is_empty(),
                    "No loci found in file: \"{}\"!",
                    fname
                );
                loci.push(locus.clone());
                // Update the Dataset
                self.loci = loci;
            } else {
                // Genotype data
                if entries_exist {
                    ensure!(
                        (i - 1) < self.entries.len(),
                        "There are too many entries in file: \"{}\" than expected!",
                        fname
                    ); // Note that we use i-1 because ith line is the header in the file
                    ensure!(
                        self.entries[i - 1].name == line[0],
                        "Entries in file: \"{}\" do not match existing Dataset! See line {}.",
                        fname,
                        i
                    ); // Again note that we use i-1 because ith line is the header in the file
                } else {
                    let entry = Entry {
                        name: line[0].to_owned(),
                        species: line[1].to_owned(),
                        group: line[2].to_owned(),
                        notes: "".to_owned(),
                    };
                    self.entries.push(entry);
                }
                let n_loci_alleles: usize = self.loci.iter().map(|x| x.col_idx.len()).sum();
                ensure!(
                    (line.len() - 3) == n_loci_alleles,
                    "Malformed genotype data in file: \"{}\" at line {}!",
                    fname,
                    i
                );
                for (j, x) in line.iter().skip(3).enumerate() {
                    let y: Vec<&str> = x.split("|").collect();
                    ensure!(
                        y.len() == 2,
                        "Malformed genotype data in file: \"{}\" at column {}. The genotype data is expected to be paired: \"h0_dosage|h1_dosage\", e.g. \"1|1\", \"2|0\" and \"3|1\"!",
                        fname,
                        j + 3
                    );
                    let h0_dosage: f32 = match y[0].parse::<f32>() {
                        Ok(x) => x,
                        Err(_e) => {
                            bail!(
                                "Failed to parse the allele dosage from the first homologous chromosome in file: \"{}\" at line {} and column {}!",
                                fname,
                                i,
                                j
                            );
                        }
                    };
                    let h1_dosage: f32 = match y[1].parse::<f32>() {
                        Ok(x) => x,
                        Err(_e) => {
                            bail!(
                                "Failed to parse the allele dosage from the second homologous chromosome in file: \"{}\" at line {} and column {}!",
                                fname,
                                i,
                                j
                            );
                        }
                    };
                    genotype_vec.push(h0_dosage);
                    genotype_vec.push(h1_dosage);
                }
            }
        }
        let n_entries: usize = self.entries.len();
        let n_loci_alleles: usize = self.loci.iter().map(|x| x.col_idx.len()).sum::<usize>();
        ensure!(
            n_entries == (genotype_vec.len() / (2 * n_loci_alleles)),
            "There are too few entries in file: \"{}\" than expected!",
            fname
        );
        // Update and check ploidy consistency across entries and loci
        let mut ploidy: usize = 0;
        for i in 0..n_entries {
            let idx_entry: usize = i * n_loci_alleles * 2;
            for locus in self.loci.iter() {
                let mut h0_dosage: usize = 0;
                let mut h1_dosage: usize = 0;
                for idx_locus_allele in locus.col_idx.iter() {
                    h0_dosage += genotype_vec[idx_entry + (2 * idx_locus_allele)] as usize;
                    h1_dosage += genotype_vec[idx_entry + (2 * idx_locus_allele) + 1] as usize;
                }
                if (i == 0) && (locus == &self.loci[0]) {
                    ploidy = h0_dosage + h1_dosage;
                } else {
                    ensure!(
                        ploidy == h0_dosage + h1_dosage,
                        "Mismatched ploidy levels in file: \"{}\" at line {} and columns {:?}!",
                        fname,
                        i + 1,
                        locus.col_idx.iter().map(|x| x + 3).collect::<Vec<usize>>()
                    );
                }
            }
        }
        self.ploidy = ploidy;
        // Update genotype data
        self.genotype_data = GpuTensor::from_vec_f32(
            ctx,
            &genotype_vec,
            &[n_entries as u32, n_loci_alleles as u32, 2u32],
            None,
            None,
        )?;
        // Note that the resulting mutated Dataset struct may be:
        // ✅ entries
        // ✅ ploidy
        // ❌ sexes
        // ✅ genome
        // ✅ loci
        // ❌ haplotype_persistence_probs
        // ❌ traits
        // ✅ genotype_data
        // ❌ phenotype_data
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::linalg::context::GpuContext;
    use std::fs::{read_to_string, remove_file, write};
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
    fn phenotype_roundtrip() {
        let ctx = context();

        let ds1 = test_dataset(&ctx);

        let path = temp_file("phenotype_roundtrip.tsv");

        ds1.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        let mut ds2 = Dataset::new(&ctx, 1).unwrap();

        ds2.read_phenotype(&ctx, path.to_str().unwrap(), "\t", false)
            .unwrap();

        assert_eq!(ds1.entries, ds2.entries);

        assert_eq!(ds1.traits, ds2.traits);

        assert_eq!(
            ds1.phenotype_data.to_vec_f32(&ctx).unwrap(),
            ds2.phenotype_data.to_vec_f32(&ctx).unwrap()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn phenotype_header_validation_fails() {
        let ctx = context();

        let path = temp_file("phenotype_bad_header.tsv");

        write(&path, "species\tname\tgroup\ttrait1\nentry\tsp\tgrp\t1.0\n").unwrap();

        let mut ds = Dataset::new(&ctx, 1).unwrap();

        assert!(
            ds.read_phenotype(&ctx, path.to_str().unwrap(), "\t", false)
                .is_err()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn phenotype_entry_mismatch_fails() {
        let ctx = context();

        let ds = test_dataset(&ctx);

        let path = temp_file("phenotype_entry_mismatch.tsv");

        ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        let mut ds2 = test_dataset(&ctx);

        ds2.entries[0].name = "__mismatch__".to_owned();

        assert!(
            ds2.read_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
                .is_err()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn overwrite_false_fails() {
        let ctx = context();

        let ds = test_dataset(&ctx);

        let path = temp_file("overwrite_false.tsv");

        ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        assert!(
            ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", false)
                .is_err()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn overwrite_true_succeeds() {
        let ctx = context();

        let ds = test_dataset(&ctx);

        let path = temp_file("overwrite_true.tsv");

        ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        assert!(
            ds.write_phenotype(&ctx, path.to_str().unwrap(), "\t", true)
                .is_ok()
        );

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

        let row = text.lines().nth(1).unwrap();

        for value in row.split('\t').skip(3) {
            assert_eq!(value.matches('|').count(), 1);
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

        let expected = format!("{}|{}", genotype[0], genotype[1]);

        assert_eq!(cols[3], expected);

        let _ = remove_file(&path);
    }

    #[test]
    fn genotype_roundtrip() {
        let ctx = context();

        let ds1 = test_dataset(&ctx);

        let path = temp_file("genotype_roundtrip.tsv");

        ds1.write_genotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        let mut ds2 = Dataset::new(&ctx, 1).unwrap();

        ds2.genome = ds1.genome.clone();

        ds2.read_genotype(&ctx, path.to_str().unwrap(), "\t", false)
            .unwrap();

        assert_eq!(ds1.entries, ds2.entries);

        assert_eq!(ds1.loci, ds2.loci);

        assert_eq!(
            ds1.genotype_data.to_vec_f32(&ctx).unwrap(),
            ds2.genotype_data.to_vec_f32(&ctx).unwrap()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn genotype_entry_mismatch_fails() {
        let ctx = context();

        let ds = test_dataset(&ctx);

        let path = temp_file("genotype_entry_mismatch.tsv");

        ds.write_genotype(&ctx, path.to_str().unwrap(), "\t", true)
            .unwrap();

        let mut ds2 = test_dataset(&ctx);

        ds2.entries[0].name = "__mismatch__".to_owned();

        assert!(
            ds2.read_genotype(&ctx, path.to_str().unwrap(), "\t", true)
                .is_err()
        );

        let _ = remove_file(&path);
    }

    #[test]
    fn genotype_invalid_header_fails() {
        let ctx = context();

        let reference = test_dataset(&ctx);

        let path = temp_file("genotype_invalid_header.tsv");

        write(
            &path,
            "name\tspecies\tgroup\tbad_header\nentry\tsp\tgrp\t1|1\n",
        )
        .unwrap();

        let mut ds = Dataset::new(&ctx, 1).unwrap();

        ds.genome = reference.genome.clone();

        assert!(
            ds.read_genotype(&ctx, path.to_str().unwrap(), "\t", false)
                .is_err()
        );

        let _ = remove_file(&path);
    }
}
