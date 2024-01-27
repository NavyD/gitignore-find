use std::{
    collections::HashSet,
    fmt::Debug,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, bail, Context, Error, Result};
use dashmap::DashSet;
use git2::{ErrorCode, Repository};
use globset::{GlobBuilder, GlobSetBuilder};
use itertools::{Either, Itertools};
use jwalk::{rayon::prelude::*, WalkDir};
use log::{debug, log_enabled, trace};
use pyo3::{
    prelude::*,
    types::{PyList, PyString},
};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn gitignore_find(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(find_ignoreds, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (path, excludes=None))]
// fn find_ignoreds(path: &PyString, excludes: Option<&PyList>) -> Result<Vec<String>> {
fn find_ignoreds(path: &PyString, excludes: Option<&PyList>) -> Result<Vec<PathBuf>> {
    let path = path.to_str()?;
    let excludes = excludes
        .map(|e| e.extract::<Vec<&str>>())
        .unwrap_or_else(|| Ok(vec![]))?;

    find(path, excludes).map(|i| i.collect::<Vec<_>>())
}

fn find<P, Q>(path: P, excludes: Q) -> Result<impl Iterator<Item = PathBuf>>
where
    P: AsRef<Path>,
    Q: IntoIterator,
    Q::Item: Into<String>,
{
    let path = path.as_ref();
    let all_paths = find_all_paths(path, excludes).unwrap();
    debug!("Finding ignoreds in {}", path.display());
    let ignoreds = DashSet::new();
    all_paths
        .par_iter()
        .filter(|p| !ignoreds.contains(*p) && p.is_dir())
        .try_for_each(|p| {
            let repo_paths = all_paths
                .iter()
                .filter(|ap| ap.starts_with(p))
                .collect_vec();
            trace!(
                "Checking if {} paths of git repo {} has ignoreds",
                repo_paths.len(),
                p.display()
            );
            match find_ignoreds_in_repo(p, repo_paths) {
                Ok(sub_ignoreds) => {
                    let sub_ignoreds = sub_ignoreds.collect_vec();
                    debug!(
                        "Found {} ignoreds paths in repo {}",
                        sub_ignoreds.len(),
                        p.display()
                    );
                    for subp in sub_ignoreds {
                        ignoreds.insert(subp.clone());
                    }
                }
                Err(e) => {
                    if let Some(git_err) = e.downcast_ref::<git2::Error>() {
                        // ignore invalid git dir
                        if matches!(git_err.code(), ErrorCode::NotFound) {
                            return Ok(());
                        }
                    }
                    return Err(e);
                }
            }
            Ok::<_, Error>(())
        })?;
    Ok(ignoreds.into_iter())
}

fn find_all_paths<P, Q>(path: P, excludes: Q) -> Result<Vec<PathBuf>>
where
    P: AsRef<Path>,
    Q: IntoIterator,
    Q::Item: Into<String>,
{
    // trace!(
    //     "Converting glob set from {} excludes patterns: {:?}",
    //     excludes.len(),
    //     excludes
    // );
    let exclude_pat = excludes
        .into_iter()
        .try_fold(GlobSetBuilder::new(), |mut gs, s| {
            let glob = GlobBuilder::new(&s.into())
                .literal_separator(true)
                .build()?;
            gs.add(glob);
            Ok::<_, Error>(gs)
        })
        .and_then(|b| b.build().map_err(Into::into))?;
    let path = path.as_ref().canonicalize()?;
    debug!("Traversing all paths under directory {}", path.display());
    let all_paths = WalkDir::new(&path)
        .sort(true)
        .process_read_dir(move |_depth, _path, _read_dir_state, children| {
            // let exclude_pat = exclude_pat.lock().unwrap();
            children.retain(|dir_ent| {
                dir_ent
                    .as_ref()
                    .map(|ent| !exclude_pat.is_match(ent.path()))
                    .unwrap_or(false)
            });
        })
        .into_iter()
        .map(|dir_ent| dir_ent.map(|e| e.path()).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    debug!("Found {} paths in {}", all_paths.len(), path.display());
    Ok(all_paths)
}

fn find_ignoreds_in_repo<P, Q, I>(path: P, repo_paths: I) -> Result<impl Iterator<Item = Q>>
where
    P: AsRef<Path> + Debug,
    Q: AsRef<Path> + Debug,
    I: IntoIterator<Item = Q>,
{
    let path = path.as_ref().canonicalize()?;
    trace!("Opening git repo in {}", path.display());
    let repo = Repository::open(&path)?;

    let (results, errors): (Vec<_>, Vec<_>) = repo_paths
        .into_iter()
        .map(|p| repo.is_path_ignored(p.as_ref()).map(|ignored| (ignored, p)))
        .partition(|r| r.is_ok());
    let errors = errors
        .into_iter()
        .map(|r| r.map_err(Into::<anyhow::Error>::into))
        .map(Result::unwrap_err)
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        bail!("Found ignore errors: {:?}", errors)
    }

    let (mut ignoreds, _) = results
        .into_iter()
        .map(Result::unwrap)
        .partition_map::<Vec<_>, Vec<_>, _, _, _>(|(ignored, p)| {
            if ignored {
                Either::Left(p)
            } else {
                Either::Right(p)
            }
        });
    trace!(
        "Found {} ignored paths in repo {}",
        ignoreds.len(),
        path.display()
    );

    // merge sub path to parent
    ignoreds.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    // TODO: 如何解决vec自引用循环中的问题
    let ignoreds_set = ignoreds
        .iter()
        .map(|p| p.as_ref().to_path_buf())
        .collect::<HashSet<_>>();
    Ok(ignoreds.into_iter().filter(move |p| {
        p.as_ref()
            .ancestors()
            // 跳过当前path
            .skip(1)
            .all(|pp| !ignoreds_set.contains(pp))
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use log::LevelFilter;

    use super::*;

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
        let path = Path::new("../lede");
        let repo_paths = find_all_paths::<_, &[String]>(path, &[])?;
        for p in repo_paths {
            println!("{}", p.display());
        }
        Ok(())
    }

    #[test]
    fn test_find_ignoreds_in_this_repo() -> Result<()> {
        let path = Path::new(".");
        let repo_paths = find_all_paths::<_, &[String]>(path, &[])?;
        let pp = repo_paths.iter().map(|p| p.to_str().unwrap()).collect_vec();
        let ignoreds = find_ignoreds_in_repo(".", repo_paths)?.collect_vec();
        assert!(!ignoreds.is_empty());
        assert_eq!(path.canonicalize()?.join("target"), ignoreds[0]);
        Ok(())
    }

    #[test]
    fn test_find() -> Result<()> {
        let path = Path::new("../lede");
        // let ignoreds = find(path, ["**/.git/**"])?.collect_vec();
        let repo_paths = find_all_paths::<_, &[String]>(path, &[])?;
        assert!(repo_paths.contains(&path.join("bin")));
        // let ignoreds = find::<_, &[String]>("../lede", &[])?.collect_vec();
        // assert!(!ignoreds.is_empty());
        // println!("{} {:?}", ignoreds.len(), ignoreds);

        Ok(())
    }
}
