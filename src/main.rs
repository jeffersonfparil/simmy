use simmy::linalg::multiplication::MatMulParams;
use simmy::run;

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    pollster::block_on(run())?;
    println!("size = {}", std::mem::size_of::<MatMulParams>());
    println!("size  = {}", std::mem::size_of::<MatMulParams>());
    println!("align = {}", std::mem::align_of::<MatMulParams>());
    Ok(())
}
