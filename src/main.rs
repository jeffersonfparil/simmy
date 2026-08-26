use simmy::linalg::addition::TensorAddParams;
use simmy::linalg::multiplication::MatrixMulParams;
use simmy::run;

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    pollster::block_on(run())?;
    println!("Matrix size = {}", std::mem::size_of::<MatrixMulParams>());
    println!("Matrix size  = {}", std::mem::size_of::<MatrixMulParams>());
    println!("Matrix align = {}", std::mem::align_of::<MatrixMulParams>());
    println!("Tensor size = {}", std::mem::size_of::<TensorAddParams>());
    println!("Tensor size  = {}", std::mem::size_of::<TensorAddParams>());
    println!("Tensor align = {}", std::mem::align_of::<TensorAddParams>());
    Ok(())
}
