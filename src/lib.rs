pub mod io;
pub mod linalg;

use anyhow::Result;
use linalg::context::GpuContext;
use linalg::tensor::GpuTensor;

use crate::io::{Data, MapperG2P};

// TODO: this were CLI arguments parsing will live...

pub async fn run() -> Result<()> {
    let ctx = pollster::block_on(GpuContext::new()).expect("Failed to create GPU context");
    println!("ctx: {}", ctx);
    let a = GpuTensor::from_vec_f32(
        &ctx,
        &(0..6).map(|x| x as f32).collect::<Vec<f32>>(),
        &[2, 3],
        None,
        None,
    )?;
    println!("a: {}", a);
    let b = GpuTensor::from_vec_f32(
        &ctx,
        &(0..12).map(|x| x as f32).collect::<Vec<f32>>(),
        &[3, 4],
        None,
        None,
    )?;
    println!("b: {}", b);
    // let ops = MatrixOps { ctx: &ctx };
    // let c = ops.multiply(&a, &b)?;
    // println!("c: {}", c);

    let n_entries: usize = 100;
    let n_chromosomes: usize = 10;
    let n_loci: usize = 1_000;
    let n_traits: usize = 2;
    let ploidy: usize = 2;
    let with_sex: bool = true;
    let seed: u64 = 4;
    let (founders, mapper): (Data, MapperG2P) = Data::sim_founders(
        &ctx,
        n_entries,
        n_chromosomes,
        n_loci,
        n_traits,
        ploidy,
        with_sex,
        seed,
    )?;
    println!("founders:\n{}", founders);
    println!("mapper:\n{}", mapper);

    println!("founders.entries:\n{:?}", founders.entries);
    println!("mapper.genome:\n{:?}", mapper.genome);

    Ok(())
}
