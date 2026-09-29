use anyhow::{Result, ensure};

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub species: String,
    pub group: String,
    pub notes: String,
}

pub fn sim_entries(n_entries: usize) -> Result<Vec<Entry>> {
    ensure!(
        n_entries > 0,
        "The number of entries need to be greater than zero!"
    );
    let mut entries: Vec<Entry> = Vec::with_capacity(n_entries);
    let n_digits: usize = format!("{}", n_entries - 1).len();
    for i in 0..n_entries {
        entries.push(Entry {
            name: format!("entry_{:0>n_digits$}", i),
            species: "".to_owned(),
            group: "".to_owned(),
            notes: "".to_owned(),
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_entries_rejects_zero() {
        assert!(sim_entries(0).is_err());
    }
}
