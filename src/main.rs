use simmy::run;

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    pollster::block_on(run())?;
    Ok(())
}
