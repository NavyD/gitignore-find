use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let cwd = std::env::current_dir().unwrap();
    let cwd = cwd.to_str().unwrap();
    // let cwd = "/home/navyd/.local/share/chezmoi";
    // let mut group = c.benchmark_group("sample-size-find");
    // group.significance_level(0.1).sample_size(500);
    c.bench_function("gitignore find", |b| {
        b.iter(|| {
            gitignore_find::find(
                [&cwd],
                [format!("{cwd}/.git/**")],
                ["**/.venv/bin/activate"],
            )
            .unwrap()
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = criterion_benchmark
}
criterion_main!(benches);
