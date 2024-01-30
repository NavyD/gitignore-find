use std::{
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Error, Result};
use globset::{GlobBuilder, GlobSetBuilder};
use ignore::gitignore::Gitignore;
use itertools::Itertools;
use jwalk::{rayon::prelude::*, WalkDir};
use log::{debug, trace};
use pyo3::{
    prelude::*,
    types::{PyList, PyString},
};

/// A Python module implemented in Rust.
#[pymodule]
fn gitignore_find(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_ignoreds, m)?)?;
    Ok(())
}

#[pyfunction]
fn find_ignoreds(path: &PyString, excludes: Option<&PyList>) -> Result<Vec<PathBuf>> {
    let path = path.to_str()?;
    let excludes = excludes
        .map(|e| e.extract::<Vec<&str>>())
        .unwrap_or_else(|| Ok(vec![]))?;

    let paths = find_paths(path, excludes)?;
    let ignoreds = find_gitignoreds(paths).sorted().collect_vec();
    Ok(ignoreds)
}

fn find_paths<'a, P, I>(path: P, excludes: I) -> Result<Vec<PathBuf>>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = &'a str>,
{
    let exclude_pat = excludes
        .into_iter()
        .try_fold(GlobSetBuilder::new(), |mut gs, s| {
            let glob = GlobBuilder::new(s).literal_separator(true).build()?;
            gs.add(glob);
            Ok::<_, Error>(gs)
        })
        .and_then(|b| b.build().map_err(Into::into))?;
    let path = path.as_ref();

    debug!("Traversing all paths in directory {}", path.display());
    WalkDir::new(path)
        .sort(true)
        .skip_hidden(false)
        .process_read_dir(move |_depth, _path, _read_dir_state, children| {
            // let exclude_pat = exclude_pat.lock().unwrap();
            if !exclude_pat.is_empty() {
                children.retain(|dir_ent| {
                    dir_ent
                        .as_ref()
                        .map(|ent| !exclude_pat.is_match(ent.path()))
                        .unwrap_or(false)
                });
            }
        })
        .into_iter()
        .map(|dir_ent| dir_ent.map(|e| e.path()).map_err(Into::into))
        .collect::<Result<Vec<_>>>()
}

fn find_gitignoreds<I, P>(paths: I) -> impl Iterator<Item = P>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path> + Send + Sync + Debug + Eq + Hash + Ord,
{
    let paths = paths.into_iter().map(Arc::new).collect_vec();
    let ignoreds = paths
        .par_iter()
        .filter(|p| p.as_ref().as_ref().ends_with(".gitignore") && p.as_ref().as_ref().is_file())
        .filter_map(|path| {
            let path = path.as_ref().as_ref();
            trace!("Loading gitignore rule from {}", path.display());
            let (gi, err) = Gitignore::new(path);
            if let Some(e) = err {
                debug!(
                    "Ignore loaded gitignore rule error in {}: {}",
                    path.display(),
                    e
                );
            }
            path.parent().map(|dir| {
                let cur_ignoreds = paths
                    .iter()
                    .filter(|p| {
                        let p = p.as_ref().as_ref();
                        p != dir && p.starts_with(dir) && gi.matched(p, p.is_dir()).is_ignore()
                    })
                    .collect_vec();
                debug!(
                    "Found {} ignoreds paths in {}",
                    cur_ignoreds.len(),
                    dir.display()
                );
                let set = cur_ignoreds
                    .iter()
                    .map(AsRef::as_ref)
                    .map(AsRef::as_ref)
                    .collect::<HashSet<_>>();
                let mergeds = cur_ignoreds
                    .iter()
                    .filter(|p| {
                        p.as_ref()
                            .as_ref()
                            .ancestors()
                            .skip(1)
                            .all(|pp| !set.contains(pp))
                    })
                    .copied()
                    .collect_vec();
                trace!(
                    "Merged {} ignored paths in {}: {:?}",
                    mergeds.len(),
                    dir.display(),
                    mergeds
                );
                mergeds
            })
        })
        .flatten()
        .map(Arc::clone)
        .collect::<HashSet<_>>();

    drop(paths);

    let ignoreds_set = ignoreds
        .iter()
        .map(|p| p.as_ref().as_ref())
        .collect::<HashSet<_>>();
    ignoreds
        .iter()
        // 合并已存在的子路径
        .filter(|p| {
            p.as_ref()
                .as_ref()
                .ancestors()
                .skip(1)
                .all(|pp| !ignoreds_set.contains(pp))
        })
        .map(Arc::clone)
        .collect::<Vec<_>>()
        .into_iter()
        // safety: paths has dropped
        .map(|p| Arc::try_unwrap(p).unwrap())
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;
    use log::LevelFilter;
    use pretty_assertions::assert_eq;

    #[ctor::ctor]
    fn init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            env_logger::builder()
                .is_test(true)
                .filter_level(LevelFilter::Info)
                .filter_module(env!("CARGO_CRATE_NAME"), LevelFilter::Trace)
                .init();
        });
    }

    #[test]
    fn test_find_all_paths() -> Result<()> {
        let path = Path::new(".");
        let paths = find_paths(path, [])?.into_iter().sorted().collect_vec();
        assert!(!paths.is_empty());
        assert!(paths.contains(&path.join("target")));
        assert!(paths.contains(&path.join(".git")));
        assert!(paths.contains(&path.join(".gitignore")));

        let expect_paths = walkdir::WalkDir::new(path)
            .into_iter()
            .map(|ent| ent.map(|e| e.into_path()).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sorted()
            .collect_vec();
        assert_eq!(paths, expect_paths);
        Ok(())
    }

    #[test]
    fn test_find_gitignoreds() -> Result<()> {
        let base = Path::new(".");
        let paths = find_paths(base, [])?;
        assert!(
            paths.iter().any(|p| p.ends_with(".gitignore")),
            "gitignore file exists"
        );
        let ignoreds = find_gitignoreds(&paths).collect_vec();
        assert!(!ignoreds.is_empty());
        assert!(ignoreds.contains(&&base.join("target")));
        assert!(ignoreds.contains(&&base.join(".venv")));

        // let base = Path::new("/home/navyd/.local/share/chezmoi");
        // let paths = find_paths(base, [])?;
        // assert!(
        //     paths.iter().any(|p| p.ends_with(".gitignore")),
        //     "gitignore file exists"
        // );
        // let ignoreds = find_gitignoreds(&paths).collect_vec();
        // assert!(!ignoreds.is_empty());
        // assert!(ignoreds.contains(&&base.join("build/.resticignore")));

        // let base = Path::new("/home/navyd/");
        // let paths = find_paths(base, [])?;
        // assert!(
        //     paths.iter().any(|p| p.ends_with(".gitignore")),
        //     "gitignore file exists"
        // );
        // let ignoreds = find_gitignoreds(&paths).collect_vec();
        // assert!(!ignoreds.is_empty());
        Ok(())
    }
}
