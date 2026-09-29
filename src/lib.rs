pub mod io;
pub mod linalg;

use anyhow::Result;
use linalg::context::GpuContext;
use linalg::tensor::GpuTensor;
// use rand::SeedableRng;
// use rand_chacha::ChaCha8Rng;
// use rand_distr::{Distribution, Uniform};

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

    // let n_entries = 5000;
    // let n_loci = 200;
    // let n_traits = 2;
    // let mut rng = ChaCha8Rng::seed_from_u64(42);
    // let range =
    //     Uniform::new(n_loci / 10, n_loci / 2).expect("Failed to initialise uniform distribution!");
    // let mut data = io::Data::new(&ctx, n_entries, 5, n_loci, n_traits, 2, false, 42).unwrap();
    // let mut genetic_params = Vec::with_capacity(n_traits);
    // let mut error_sds = Vec::with_capacity(n_traits);
    // for i in 0..n_traits {
    //     genetic_params.push((range.sample(&mut rng), i as f32, 1.0));
    //     error_sds.push(1.0);
    // }
    // let effects = data.sim_allele_effects(&ctx, &genetic_params, 123).unwrap();
    // data.sim_phenotypes(&effects, &ctx, &error_sds, 456)
    //     .unwrap();
    // let b: Vec<f32> = effects.to_vec_f32(&ctx)?;
    // let x: Vec<f32> = data.genotype_data.to_vec_f32(&ctx)?;
    // let y: Vec<f32> = data.phenotype_data.to_vec_f32(&ctx)?;
    // println!("b: [{}, {}, {}, ..., {}]", b[0], b[1], b[2], b[b.len() - 1]);
    // println!("x: [{}, {}, {}, ..., {}]", x[0], x[1], x[2], x[x.len() - 1]);
    // println!("y: [{}, {}, {}, ..., {}]", y[0], y[1], y[2], y[y.len() - 1]);

    Ok(())
}
