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
    let a = GpuTensor::from_f32(
        &ctx,
        &(0..6).map(|x| x as f32).collect::<Vec<f32>>(),
        &[2, 3],
        None,
        None,
    )?;
    println!("a: {}", a);
    let b = GpuTensor::from_f32(
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
    Ok(())
}
