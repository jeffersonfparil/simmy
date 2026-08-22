pub mod linalg;

use anyhow::Result;
use linalg::context::GpuContext;
use linalg::tensor::GpuTensor;
use linalg::kernel::GpuKernel;
use linalg::operations::MatrixOps;

// TODO: this were CLI arguments parsing will live...

pub async fn run() -> Result<()> {
    let ctx = pollster::block_on(GpuContext::new()).expect("Failed to create GPU context");
    println!("ctx: {}", ctx);
    let a = GpuTensor::from_f32(
        &ctx,
        vec![2,3],
        &(0..6).map(|x| x as f32).collect::<Vec<f32>>(),
    )?;
    println!("a: {}", &a);
    let b = GpuTensor::from_f32(
        &ctx,
        vec![3, 4],
        &(0..12).map(|x| x as f32).collect::<Vec<f32>>(),
    )?;
    println!("b: {}", &b);
    let ops = MatrixOps { ctx: &ctx };
    let c = ops.multiply(&a, &b)?;
    println!("c: {}", &c);
    Ok(())
}