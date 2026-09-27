pub mod io;
pub mod linalg;

use anyhow::Result;
use linalg::context::GpuContext;
use linalg::tensor::GpuTensor;
// use linalg::operations::MatrixOps;
// use linalg::kernel::GpuKernel;

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

    let n_traits = 2;
    let mut data = io::Data::new(&ctx, 10, 2, 20, n_traits, 2, false, 42)?;
    let params = [(5, 0.0, 1.0, 1.0), (3, 0.0, 1.0, 1.0)];
    let _effects_tensor = data.sim_phenotypes(&ctx, &params, 123)?;

    Ok(())
}
