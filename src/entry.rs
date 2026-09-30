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
    use std::collections::HashSet;

    #[test]
    fn sim_entries_rejects_zero() {
        assert!(sim_entries(0).is_err());
    }

    #[test]
    fn sim_entries_returns_requested_number() {
        let entries = sim_entries(37).unwrap();

        assert_eq!(entries.len(), 37);
    }

    #[test]
    fn sim_entries_initialises_empty_metadata() {
        let entries = sim_entries(10).unwrap();

        assert!(
            entries
                .iter()
                .all(|e| { e.species.is_empty() && e.group.is_empty() && e.notes.is_empty() })
        );
    }

    #[test]
    fn sim_entries_generates_unique_names() {
        let entries = sim_entries(1000).unwrap();

        let names: HashSet<_> = entries.iter().map(|e| &e.name).collect();

        assert_eq!(names.len(), entries.len());
    }

    #[test]
    fn sim_entries_is_deterministic() {
        let a = sim_entries(100).unwrap();
        let b = sim_entries(100).unwrap();

        let names_a: Vec<_> = a.iter().map(|e| e.name.as_str()).collect();

        let names_b: Vec<_> = b.iter().map(|e| e.name.as_str()).collect();

        assert_eq!(names_a, names_b);
    }

    #[test]
    fn sim_entries_single_entry_has_expected_name() {
        let entries = sim_entries(1).unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "entry_0");
    }

    #[test]
    fn sim_entries_uses_expected_names_for_small_dataset() {
        let entries = sim_entries(5).unwrap();

        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();

        assert_eq!(
            names,
            vec!["entry_0", "entry_1", "entry_2", "entry_3", "entry_4",]
        );
    }

    #[test]
    fn sim_entries_zero_pads_two_digits() {
        let entries = sim_entries(100).unwrap();

        assert_eq!(entries[0].name, "entry_00");
        assert_eq!(entries[9].name, "entry_09");
        assert_eq!(entries[10].name, "entry_10");
        assert_eq!(entries[99].name, "entry_99");
    }

    #[test]
    fn sim_entries_zero_pads_three_digits() {
        let entries = sim_entries(1000).unwrap();

        assert_eq!(entries[0].name, "entry_000");
        assert_eq!(entries[9].name, "entry_009");
        assert_eq!(entries[99].name, "entry_099");
        assert_eq!(entries[999].name, "entry_999");
    }

    #[test]
    fn sim_entries_first_and_last_names_are_correct() {
        let entries = sim_entries(123).unwrap();

        assert_eq!(entries.first().unwrap().name, "entry_000");
        assert_eq!(entries.last().unwrap().name, "entry_122");
    }
}
