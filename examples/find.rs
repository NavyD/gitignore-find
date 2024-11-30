use anyhow::Result;
use clap::Parser;
use itertools::Itertools;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(default_value = ".")]
    paths: Vec<String>,
    #[arg(short, long)]
    excludes: Vec<String>,
}

fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    env_logger::builder()
        .is_test(true)
        .format_timestamp_millis()
        .filter_level(log::LevelFilter::Error)
        .filter_module("gitignore_find", log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    let ignoreds = gitignore_find::find(&args.paths, &args.excludes)?;
    println!(
        "{}",
        ignoreds.iter().map(|p| p.display().to_string()).join("\n")
    );
    Ok(())
}
