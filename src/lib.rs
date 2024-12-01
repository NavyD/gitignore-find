use core::panic;
use std::{
    fmt::Debug,
    hash::Hash,
    path::{Path, PathBuf},
    rc::Rc,
};

use anyhow::{Error, Result};
use globset::{GlobBuilder, GlobSetBuilder};
use ignore::gitignore::Gitignore;
use itertools::Itertools;
use jwalk::{rayon::prelude::*, WalkDir};
#[allow(unused_imports)]
use log::{debug, log_enabled, trace, warn};
use pyo3::{
    prelude::*,
    types::{PySequence, PyString},
};
use sha2::{Digest, Sha256};

#[cfg(not(debug_assertions))]
use hashbrown::{HashMap, HashSet};
#[cfg(debug_assertions)]
use std::collections::{HashMap, HashSet};

#[cfg(all(
    not(feature = "dhat-heap"),
    // 在gnu linux非x86平台可能会构建失败
    not(all(
        target_os = "linux",
        target_env = "gnu",
        not(any(target_arch = "x86_64", target_arch = "x86"))
    ))
))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const PARALLEL_PATH_SIZE: usize = 200000;

/// A Python module implemented in Rust.
#[cfg(not(tarpaulin_include))]
#[pymodule]
fn gitignore_find(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::Logger::new(m.py(), pyo3_log::Caching::LoggersAndLevels)?
        // 仅启用当前模块的log否则可能有其它模块如ignore
        .filter(log::LevelFilter::Warn)
        .filter_target(env!("CARGO_CRATE_NAME").to_owned(), log::LevelFilter::Trace)
        .install()
        .unwrap();
    m.add_function(wrap_pyfunction!(find_ignoreds, m)?)?;
    Ok(())
}

#[cfg(not(tarpaulin_include))]
#[pyfunction]
#[pyo3(signature = (paths, excludes=None))]
fn find_ignoreds(
    paths: &Bound<'_, PyAny>,
    excludes: Option<&Bound<'_, PySequence>>,
) -> Result<Vec<PathBuf>> {
    // paths支持str|sequence
    let paths = if let Ok(path) = paths.downcast::<PyString>() {
        vec![path.to_string()]
    } else {
        // NOTE: 使用downcast优于extract，但downcast需要手动一层层的转换处理err太麻烦，其错误不兼容anyhow error
        paths.extract::<Vec<String>>()?
    };
    let excludes = excludes
        .map(|e| e.extract::<Vec<String>>())
        .unwrap_or_else(|| Ok(vec![]))?;
    find(paths, excludes)
}

#[cfg(not(tarpaulin_include))]
pub fn find(
    paths: impl IntoIterator<Item: AsRef<Path> + Clone + Debug>,
    excludes: impl IntoIterator<Item: AsRef<str> + Clone + Debug>,
) -> Result<Vec<PathBuf>> {
    let (paths, excludes) = (
        paths.into_iter().collect_vec(),
        excludes.into_iter().collect_vec(),
    );
    if log_enabled!(log::Level::Debug) {
        debug!(
            "Finding git ignored paths with exclude globs {:?} in {} paths: {:?}",
            excludes,
            paths.len(),
            paths
        );
    }
    let exclude_pat = GlobPathPattern::new(excludes)?;
    let ignoreds = paths
        .iter()
        .map(|path| find_gitignoreds(path, &exclude_pat))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.into_iter().flatten().collect_vec())?;

    debug!("Found {} ignored paths for {:?}", ignoreds.len(), paths);
    Ok(ignoreds)
}

fn find_all_paths_iter(path: impl AsRef<Path>) -> impl Iterator<Item = PathBuf> {
    trace!("Traversing paths in directory {}", path.as_ref().display());
    WalkDir::new(path)
        .skip_hidden(false)
        .process_read_dir(move |_depth, _path, _read_dir_state, children| {
            children.retain(|dir_ent| {
                dir_ent
                    .as_ref()
                    // 忽略.git目录
                    .map(|ent| {
                        ent.path().file_name().and_then(|s| s.to_str()) != Some(".git")
                            || !ent.path().is_dir()
                    })
                    .unwrap_or(false)
            });
        })
        .into_iter()
        // .map(|dir_ent| dir_ent.map(|e| e.path()).map_err(Into::into))
        .filter_map(move |p| match p {
            Ok(p) => Some(p.path()),
            Err(e) => {
                warn!("Ignore to get path error: {}", e);
                None
            }
        })
}

trait PathPattern {
    fn is_match<P>(&self, p: P) -> bool
    where
        P: AsRef<Path>;

    fn is_empty(&self) -> bool;
}

struct GlobPathPattern {
    set: globset::GlobSet,
    patterns: Vec<String>,
}

impl GlobPathPattern {
    fn new(pats: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self> {
        let (set, patterns) = pats.into_iter().try_fold(
            (GlobSetBuilder::new(), Vec::new()),
            |(mut gs, mut patterns), s| {
                let glob = GlobBuilder::new(s.as_ref())
                    .literal_separator(true)
                    .build()?;
                gs.add(glob);
                patterns.push(s.as_ref().to_string());
                Ok::<_, Error>((gs, patterns))
            },
        )?;
        let set = set.build()?;
        Ok(Self { set, patterns })
    }
}

impl PathPattern for GlobPathPattern {
    fn is_match<P>(&self, p: P) -> bool
    where
        P: AsRef<Path>,
    {
        self.set.is_match(p)
    }

    fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

impl Debug for GlobPathPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobPathPattern")
            // .field("set", &self.set)
            .field("patterns", &self.patterns)
            .finish()
    }
}

impl<T: AsRef<str>> TryFrom<&[T]> for GlobPathPattern {
    type Error = Error;

    fn try_from(value: &[T]) -> std::result::Result<Self, Self::Error> {
        GlobPathPattern::new(value)
    }
}

type PathDigestTy = [u8; 32];

/// 对于path中所有的`.gitignore`，返回所有被其忽略的文件和目录。
///
/// ## 性能
///
/// 由于现代OS都有文件缓存，短时间多次遍历目录速度较快，所以不需要保存扫描的目录路径，也
/// 减少了大量内存使用
///
/// ## 合并目录
///
/// 由于[`Gitignore`]只匹配文件夹而不会对其中的文件匹配如`.venv`规则不会匹配`.venv/bin/test.sh`
/// 会导致忽略的路径不包含子路径。
///
/// 一个简单的实现，在检查.gitignore时，由于cur_ignoreds不包含子路径，在排除路径时需要获取所有路径，
/// 再从ignoreds中移除被排除的路径与父路径，这个路径会在后面被合并不会导致完整性问题，但存在性能问题，
/// 一旦cur_ignoreds包含的所有路径稍大会消耗的时间指数级上升，不可取
///
/// ### 当前实现
///
/// 遍历获取所有gitignores文件，所有忽略ignoreds文件，由于[`Gitignore`]的特性，被忽略的目录不会存在子路径，
/// 但由于需要排除路径，所有忽略的子目录都需要存在再用于排除，一旦被排除的路径存在ignoreds父路径中，则其所有父路径
/// 都需要移除以避免后续合并时被作为存在父中的子路径被忽略了。
///
/// 在合并时检查：
///
/// 1. 当前路径p与.gitignore所在的目录 子路径一致 且 p的父路径不存在于ignoreds 时，表示这个路径p应该保留。
///     当前路径p存在于ignored_subpaths但无subpaths时由于[`Gitignore`]特性表示所有子路径均被忽略
/// 1. 当前路径中的任意父路径存在时，其父路径不被包含在gitignore所在目录中 或 包含的gitignore目录的子路径一致，
///     表示这个路径可以忽略
///
/// 算法复杂度为O(N)最大为排序O(N log N)，使用由于并发获取路径还是会保存所有路径导致内存较大
///
/// 在比较子路径时如果数量较大时会导致内存和比较时消耗过大，可以使用hash排序生成digest比较，
/// 但在ignoreds数量在10万级以下基本是负收益
fn find_gitignoreds<'a>(
    path: impl AsRef<Path>,
    exclude_pat: &(impl PathPattern + Debug + 'a),
) -> Result<impl Iterator<Item = PathBuf>> {
    let path = path.as_ref();

    debug!("Finding all paths in {}", path.display());
    let paths = find_all_paths_iter(path).collect::<Vec<_>>();
    trace!("Found {} paths for {}", paths.len(), path.display());

    let gitignores = paths
        .iter()
        .filter(|path| {
            path.file_name().and_then(|s| s.to_str()) == Some(".gitignore") && path.is_file()
        })
        .map(|p| {
            let (gi, err) = Gitignore::new(p);
            if let Some(e) = err {
                warn!("Ignore load gitignore rule error in {}: {}", p.display(), e);
            }
            // NOTE: 不要使用gi.path()。如果是相对路径会导致`./a/.gitignore`变为`a`导致后续路径不一致
            (
                p.parent()
                    .map(ToOwned::to_owned)
                    // SAFETY: gitignore file always has parent path
                    .unwrap_or_else(|| panic!("Not found parent of path {}", p.display())),
                gi,
            )
        })
        .collect::<HashMap<_, _>>();
    #[cfg(debug_assertions)]
    trace!(
        "Found {} .gitignore files: {:?}",
        gitignores.len(),
        gitignores.keys().collect_vec(),
    );

    debug!(
        "Finding ignored paths with {} gitignores and exclude pattern {:?} in {}",
        gitignores.len(),
        exclude_pat,
        path.display(),
    );
    let take_parent_path = |p: &&Path| *p != path;
    let filter_ignored = |p: &PathBuf| {
        let keep = p
            .ancestors()
            .take_while_inclusive(take_parent_path)
            // 仅使用首个gitignore文件匹配
            .find_map(|pp| gitignores.get(pp))
            .map(|gi| {
                // 提前检查当前路径是否为gitignore目录
                if gi.path() == p {
                    // gi目录路径必为true
                    gi.matched(p, true).is_ignore()
                } else {
                    p.ancestors()
                        // 不包含gitignore目录 避免其父路径可能由于`*`规则导致忽略
                        .take_while(|newpp| *newpp != gi.path())
                        .any(|newp| {
                            // p的父路径必为目录true
                            let v = gi.matched(newp, newp != p || newp.is_dir()).is_ignore();
                            #[cfg(test)]
                            if newp.ancestors().any(|p| p.ends_with("build")) {
                                print!("");
                            }
                            v
                        })
                }
            })
            .unwrap_or(false);
        #[cfg(test)]
        print!("");
        keep
    };
    let ignoreds = if paths.len() < PARALLEL_PATH_SIZE {
        paths
            .into_iter()
            .filter(filter_ignored)
            .map(Rc::new)
            .collect::<Vec<_>>()
    } else {
        paths
            .into_par_iter()
            .filter(filter_ignored)
            .collect::<Vec<_>>()
            .into_iter()
            .map(Rc::new)
            .collect::<Vec<_>>()
    };
    debug!(
        "Found {} ignored paths for all paths {:?}{}",
        ignoreds.len(),
        path,
        if cfg!(all(debug_assertions, test)) {
            format!(": {:?}", ignoreds.iter().map(|p| p.display()).collect_vec())
        } else {
            String::new()
        }
    );
    // 如果一个路径被排除则其所有存在的父路径应该被移除，避免后续被合并到父路径中
    let ignoreds = if !exclude_pat.is_empty() {
        trace!(
            "Excluding {} paths using glob pattern: {:?}",
            ignoreds.len(),
            exclude_pat
        );
        let set = ignoreds
            .iter()
            // 升序先处理父路径后检查子路径时移除存在的父路径
            .sorted_by_cached_key(|p| {
                (
                    // std::cmp::Reverse(p.ancestors().count()),
                    p.ancestors().count(),
                    p.file_name()
                        .and_then(|s| s.to_str())
                        .map(ToOwned::to_owned),
                )
            })
            .fold(HashSet::new(), |mut set, p| {
                let p = p.as_path();
                if exclude_pat.is_match(p) {
                    // 移除所有存在的父路径
                    p.ancestors()
                        .skip(1)
                        .take_while_inclusive(take_parent_path)
                        .for_each(|pp| {
                            set.remove(pp);
                        });
                } else {
                    set.insert(p);
                }
                set
            });
        let v = ignoreds
            .iter()
            .filter(|p| set.contains(p.as_path()))
            .cloned()
            .collect_vec();
        drop(set);
        drop(ignoreds);
        v
    } else {
        ignoreds
    };

    // 用于检查父路径是否存在
    let ignoreds_set = ignoreds.iter().map(|p| p.as_path()).collect::<HashSet<_>>();
    // 用于检查子路径是否一致
    let ignored_subpaths = {
        trace!("Getting sub paths from {} ignoreds paths", ignoreds.len());
        let subpaths = get_one_level_subpaths(ignoreds.iter().map(|p| p.as_path()));

        #[cfg(debug_assertions)]
        type MapTy = HashSet<PathBuf>;
        #[cfg(not(debug_assertions))]
        type MapTy = PathDigestTy;
        fn to_digest<'a>(
            (p, paths): (&'a Path, Option<HashSet<&'a Path>>),
        ) -> (&'a Path, Option<MapTy>) {
            (
                p,
                #[cfg(debug_assertions)]
                paths.map(|paths| {
                    paths
                        .into_iter()
                        .map(ToOwned::to_owned)
                        .collect::<HashSet<_>>()
                }),
                #[cfg(not(debug_assertions))]
                paths.map(gen_digest),
            )
        }

        if subpaths.len() < PARALLEL_PATH_SIZE {
            subpaths
                .into_iter()
                .map(to_digest)
                .collect::<HashMap<_, _>>()
        } else {
            subpaths
                .into_par_iter()
                .map(to_digest)
                .collect::<HashMap<_, _>>()
        }
    };
    trace!(
        "Traversing all sub paths of {} .gitignore paths{}",
        gitignores.len(),
        if cfg!(all(debug_assertions, test)) {
            format!(
                ": {:?}",
                gitignores
                    .keys()
                    .map(|p| p.display().to_string())
                    .collect_vec()
            )
        } else {
            String::new()
        }
    );
    #[cfg(not(debug_assertions))]
    let gitignores_it = gitignores.par_keys();
    #[cfg(debug_assertions)]
    let gitignores_it = gitignores.keys().par_bridge();
    // 获取gitignore目录对应的一层实际子路径  用于与ignoreds子路径对比是否一致
    let gitignore_subpaths = gitignores_it
        .map(|p| {
            let p = p.as_path();
            // NOTE: 当ignored_subpaths的paths=None时表示这个p对应子路径不存在
            // 或所有路径均被忽略了  保存一致方便比较
            if ignored_subpaths
                .get(p)
                .map_or(false, |paths| paths.is_none())
            {
                return Ok((p, None));
            }
            let paths = std::fs::read_dir(p)?
                .map_ok(|p| p.path())
                .collect::<Result<HashSet<_>, _>>()?;
            #[cfg(debug_assertions)]
            let v = paths;
            #[cfg(not(debug_assertions))]
            let v = gen_digest(paths);
            Ok::<_, Error>((p, Some(v)))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    debug!("Merging {} ignored paths", ignoreds.len());
    let mergeds = ignoreds
        .iter()
        .filter(|p| {
            let p = p.as_path();
            let mut ances = p.ancestors().skip(1).take_while_inclusive(take_parent_path);
            #[cfg(all(debug_assertions, test))]
            if p.ancestors().any(|pp| pp.ends_with("build")) {
                print!("")
            }
            // 被忽略的当前路径是 `.gitginore`文件所在的目录
            // 如.gitignore文件内容是`*``!xxx/`时其目录会被忽略但内还有目录未被忽略
            let keep = if let Some(p_gitignore_subpaths) = gitignore_subpaths.get(p) {
                // 仅当.gitignore所在的目录 子路径一致 且 其父路径不存在 时保留
                // NOTE: 当p存在于ignored_subpaths但无subpaths时表示所有子路径均被忽略
                // 这是由于gitignore.is_match()匹配行为导致的：对于指定gitignore规则如`.venv`不会匹配`.venv/bin`
                Some(p_gitignore_subpaths) == ignored_subpaths.get(p)
                    && ances.all(|pp| !ignoreds_set.contains(pp))
            } else {
                // 当前路径中的任意父路径 存在 且 其父路径不被包含在gitignore所在目录中
                // 或 包含的gitignore目录的子路径一致，表示这个路径可以忽略
                !ances.any(|pp| {
                    ignoreds_set.contains(pp)
                        && gitignore_subpaths
                            .get(pp)
                            // is_none || ==
                            .map_or(true, |pp_gitignore_subpaths| {
                                Some(pp_gitignore_subpaths) == ignored_subpaths.get(pp)
                            })
                })
            };
            #[cfg(all(debug_assertions, test))]
            trace!(
                "Path {} is keep={} with p_gitignore_subpaths={:?} p_ignored_subpaths={:?}",
                p.display(),
                keep,
                gitignore_subpaths.get(p),
                ignored_subpaths.get(p)
            );
            keep
        })
        .cloned()
        .collect::<Vec<_>>();

    drop(ignoreds_set);
    drop(ignored_subpaths);
    drop(gitignore_subpaths);
    drop(ignoreds);
    // SAFETY: All references have been dropped
    Ok(mergeds.into_iter().map(|p| Rc::try_unwrap(p).unwrap()))
}

#[allow(dead_code)]
fn gen_digest(paths: impl IntoIterator<Item: AsRef<Path> + Ord>) -> PathDigestTy {
    paths
        .into_iter()
        .sorted_unstable()
        .fold(Sha256::new(), |d, p| {
            d.chain_update(p.as_ref().display().to_string())
        })
        .finalize()
        .into()
}

/// 获取所有路径分别对应的一层子路径paths。
///
/// # 实现
///
/// 通过一次遍历对当前路径的父路径添加当前路径
fn get_one_level_subpaths<'a, P>(
    paths: impl IntoIterator<Item = &'a P>,
) -> HashMap<&'a Path, Option<HashSet<&'a P>>>
where
    P: AsRef<Path> + Hash + Eq + 'a + ?Sized,
{
    paths.into_iter().fold(
        HashMap::<&Path, Option<HashSet<_>>>::new(),
        |mut path_subpaths, p| {
            // 为所有路径默认添加子路径为None
            let _subpaths = path_subpaths.entry(p.as_ref()).or_default();
            #[cfg(all(debug_assertions, test))]
            trace!(
                "Getting {} subpaths of {}: {:?}",
                _subpaths.as_ref().map_or(0, |v| v.len()),
                p.as_ref().display(),
                _subpaths.as_ref().map(|v| v
                    .iter()
                    .map(|p| p.as_ref().display().to_string())
                    .collect_vec()),
            );
            // 为其父路径添加p作为子路径
            if let Some((_pp, subpaths)) = p
                .as_ref()
                .parent()
                // 相对路径的顶级路径为 空 时，保留Key即可 可能为空 或 保留之前存在了子路径
                .filter(|pp| !pp.to_string_lossy().is_empty())
                .and_then(|pp| {
                    path_subpaths
                        .entry(pp)
                        // NOTE: 之前插入了None如果此时不修改会导致key已存在一直为None
                        .and_modify(|v| {
                            if v.is_none() {
                                *v = Some(HashSet::new())
                            }
                        })
                        .or_insert_with(|| Some(HashSet::new()))
                        .as_mut()
                        .map(|v| (pp, v))
                })
            {
                #[cfg(all(debug_assertions, test))]
                trace!(
                    "Adding {} to {}'s {} subpaths: {:?}",
                    p.as_ref().display(),
                    _pp.display().to_string(),
                    subpaths.len(),
                    subpaths
                        .iter()
                        .map(|p| p.as_ref().display().to_string())
                        .collect_vec()
                );
                subpaths.insert(p);
            }
            path_subpaths
        },
    )
}

/// 生成所有路径对应的 递归所有子路径 的digests。
///
/// 如果一个路径没有子路径则`digest=None`
///
/// ## 实现
///
/// 排序后自低向上查找每个路径对应的子路径。对于每个路径，使用子路径的digest计算digest，
/// 不会真的检查递归的子路径，算法复杂度是O(N)
#[allow(dead_code)]
fn gen_all_subpath_digests<P>(
    paths: impl IntoIterator<Item = P>,
) -> impl Iterator<Item = (P, Option<PathDigestTy>)>
where
    P: AsRef<Path> + Ord + Hash + Eq + Debug,
{
    let paths = {
        let mut v = paths.into_iter().map(Rc::new).collect_vec();
        v.sort_by_cached_key(|p| {
            let p = p.as_ref().as_ref();
            (
                std::cmp::Reverse(p.ancestors().count()),
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(ToOwned::to_owned),
            )
        });
        v
    };

    #[cfg(debug_assertions)]
    trace!(
        "Getting one level subpaths in {} paths: {:?}",
        paths.len(),
        paths
    );
    // 保存路径对应的子路径  如果一个路径没有子路径则为空
    let one_level_subpaths = get_one_level_subpaths(paths.iter().map(|p| p.as_ref()));
    #[cfg(debug_assertions)]
    trace!(
        "Got one level subpaths {} paths: {:?}",
        one_level_subpaths.len(),
        one_level_subpaths
    );

    type DigestTy = Sha256;
    let subpath_digests = paths.iter().fold(
        HashMap::<_, Option<PathDigestTy>>::new(),
        |mut subpath_digests, path| {
            let p = path.as_ref().as_ref();
            let subpaths = one_level_subpaths
                .get(p)
                // SAFETY: paths的所有键都存在于one_tier_subpaths中
                .unwrap_or_else(|| panic!("Not found subpaths for path {}", p.display()));
            #[cfg(debug_assertions)]
            trace!(
                "Getting digest from {} next subpaths={:?} in current path {}",
                subpaths.as_ref().map_or(0, |s| s.len()),
                subpaths,
                p.display()
            );
            // 当前路径无子路径
            let p_digest = if let Some(subpaths) = subpaths {
                // 当前存在子路径
                let p_digest: PathDigestTy = subpaths
                    .iter()
                    .fold(DigestTy::new(), |p_digest, subp| {
                        // SAFETY: 降序保证了子路径先于父路径计算digest，所以当前路径的子路径一定是计算过的 或 无子路径
                        if let Some(subp_digest) = subpath_digests.get(*subp).unwrap_or_else(|| {
                            panic!(
                                "Not found digest for sub path {} of path {}",
                                subp.as_ref().display(),
                                p.display()
                            )
                        }) {
                            p_digest.chain_update(subp_digest)
                        } else {
                            // 当前路径不存在子路径digest时如 `/a/b/1.txt`中的b子路径`1.txt`无子路径即不会
                            // 有digest，在计算到b路径的subpaths时为空，此时使用1.txt路径计算
                            let d: PathDigestTy = DigestTy::new()
                                .chain_update(subp.as_ref().display().to_string())
                                .finalize()
                                .into();
                            p_digest.chain_update(d)
                        }
                    })
                    .finalize()
                    .into();
                Some(p_digest)
            } else {
                None
            };
            subpath_digests.insert(path.clone(), p_digest);
            subpath_digests
        },
    );

    drop(one_level_subpaths);
    drop(paths);
    subpath_digests
        .into_iter()
        .map(|(k, v)| (Rc::try_unwrap(k).unwrap(), v))
}

#[cfg(test)]
mod tests {
    use std::{borrow::Borrow, sync::Once};

    use super::*;
    use log::LevelFilter;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[ctor::ctor]
    fn init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            env_logger::builder()
                .is_test(true)
                .format_timestamp_millis()
                .filter_level(LevelFilter::Info)
                .filter_module(env!("CARGO_CRATE_NAME"), LevelFilter::Trace)
                .init();
        });
    }

    /// 从repo_dir中创建相关文件与.gitignore文件，返回repo_dir所有文件与目录
    fn mock_git_paths<IP, II, IIP, Q>(
        gitignore_items: impl IntoIterator<Item = (IP, II)>,
        paths: impl IntoIterator<Item = Q>,
        repo_dir: impl AsRef<Path>,
    ) -> impl Iterator<Item = PathBuf>
    where
        Q: AsRef<Path>,
        IP: AsRef<Path>,
        II: Borrow<[IIP]>,
        IIP: AsRef<Path>,
    {
        let repo_dir = repo_dir.as_ref();

        // write .gitignore file
        let gitignore_items = gitignore_items.into_iter().collect_vec();
        if !gitignore_items.is_empty() {
            for (path, items) in gitignore_items {
                let gitignore_path = repo_dir.join(path);
                if let Some(p) = gitignore_path.parent() {
                    std::fs::create_dir_all(p).unwrap();
                }
                assert!(
                    gitignore_path.ends_with(".gitignore"),
                    "not found .gitignore suffix for {}",
                    gitignore_path.display()
                );
                std::fs::write(
                    &gitignore_path,
                    items
                        .borrow()
                        .iter()
                        .map(|i| i.as_ref().display())
                        .join("\n"),
                )
                .unwrap();
            }
        }

        // write other files
        for p in paths {
            let p = p.as_ref();
            assert!(
                !p.is_absolute(),
                "Found absolute path {} argument for git repo {}",
                p.display(),
                repo_dir.display()
            );
            let repo_file = repo_dir.join(p);
            if let Some(pp) = repo_file.parent() {
                std::fs::create_dir_all(pp).unwrap();
            }
            std::fs::write(&repo_file, format!("{}", repo_file.display())).unwrap();
        }

        // read all paths
        WalkDir::new(repo_dir)
            .skip_hidden(false)
            .into_iter()
            .map(|entry| entry.map(|e| e.path().to_owned()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
    }

    #[rstest]
    #[case::all_empty([], [], [], [])]
    #[case::no_gitignore_and_no_excludes(["1.txt", ".env", ".envrc"], [], [], [])]
    #[case::no_excludes(["1.txt", ".env", ".envrc"], [(".gitignore", &[".env*"] as &[&str])], [], [".env", ".envrc"])]
    #[case::no_gitignore(["1.txt", ".env", ".envrc"], [], ["**/.env"], [])]
    #[case::exclude_env(
        ["1.txt", ".env", ".envrc"],
        [(".gitignore", &[".env*"] as &[&str])],
        ["**/.env"],
        [".envrc"]
    )]
    #[case::nest_excludes(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth"],
        [(".gitignore", &[".env*", ".venv"]as &[&str])],
        ["**/.env", "**/.venv/**", "**/.venv"],
        [".envrc"]
    )]
    #[case::nest_excludes(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        [(".gitignore", &[".env*", ".venv"] as &[&str])],
        ["**/.env", "**/.venv/bin", "**/.venv/bin/**"],
        [".envrc", ".venv/lib", ".venv/pyvenv.cfg"],
    )]
    #[case::nest_excludes_without_globself(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        [(".gitignore", &[".env*", ".venv"] as &[&str])],
        ["**/.env", "**/.venv/bin/**"],
        [".envrc", ".venv/lib", ".venv/pyvenv.cfg"],
    )]
    #[case::nest_gitignores(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg", "build/a.txt", "build/1.txt"],
        [(".gitignore", &[".env*", ".venv"] as &[&str]), ("build/.gitignore", &["*", "!a.txt"])],
        ["**/.env", "**/.venv/bin/**"],
        [".envrc", ".venv/lib", ".venv/pyvenv.cfg", "build/1.txt", "build/.gitignore"],
    )]
    fn test_find_gitignoreds<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] gitignore_items: impl IntoIterator<Item = (&'a str, &'a [&'a str])>,
        #[case] excludes: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let tmpdir = tempfile::tempdir().unwrap();
        let paths = paths.into_iter().collect_vec();
        // NOTE: tmpdir非引用会导致提前移除目录
        let git_paths = mock_git_paths(gitignore_items, &paths, &tmpdir).collect::<Vec<PathBuf>>();
        for p in &git_paths {
            assert!(p.exists(), "path {} is not exists", p.display());
        }
        let ignoreds = find_gitignoreds(&tmpdir, &GlobPathPattern::new(excludes)?)?
            .sorted_unstable()
            .collect_vec();
        let expected = expected
            .into_iter()
            .map(|s| tmpdir.path().join(s))
            .sorted()
            .collect::<Vec<_>>();
        assert_eq!(ignoreds, expected);
        Ok(())
    }

    #[rstest]
    #[case(
        ["/home/navyd/.local/share/chezmoi"],
        // ["**/.git/**", "**/.venv/**", "**/target/**"],
        ["**/.git/**", "**/.venv/**", "**/target/**", "**/*cache/**"],
        [
            "./.venv",
            "./flamegraph.svg",
            "./perf.data",
            "./target",
            "./perf.data.old",
        ]
    )]
    #[case(
        ["."],
        ["**/.git", "**/.cargo", "**/.vscode*"],
        [
            "./.venv",
            "./flamegraph.svg",
            "./perf.data",
            "./perf.data.old",
        ]
    )]
    #[ignore = "实机测试环境可能不一致"]
    fn test_integration_find_ignoreds_in_current_repo<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] excludes: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = &'a str>,
    ) {
        fn path_sort_asc_fn(a: impl AsRef<Path>, b: impl AsRef<Path>) -> std::cmp::Ordering {
            let (a, b) = (a.as_ref(), b.as_ref());
            a.ancestors()
                .count()
                .cmp(&b.ancestors().count())
                .then_with(|| a.cmp(b))
        }
        let mut ignoreds = find(paths, excludes).unwrap();
        // let mut ignoreds = find(["."], excludes, exclude_ignoreds).unwrap();
        // let mut ignoreds = find(["."], excludes, exclude_ignoreds).unwrap();
        // ignoreds.sort_unstable_by_key(|p| (std::cmp::Reverse(p.ancestors().count()), p));
        ignoreds.sort_unstable_by(|a, b| path_sort_asc_fn(a, b));
        let expected = expected
            .into_iter()
            .map(PathBuf::from)
            .sorted_unstable_by(|a, b| path_sort_asc_fn(a, b))
            .collect::<Vec<PathBuf>>();
        assert_eq!(ignoreds, expected);
    }

    #[rstest]
    #[case(
        ["a", "build", "build/1.txt", "build/2.txt"],
        [
            ("a", &[] as &[_]),
            ("build", &["build/1.txt", "build/2.txt"]),
            ("build/1.txt", &[]),
            ("build/2.txt", &[]),
        ]
    )]
    fn test_get_one_level_subpaths<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = (&'a str, &'a [&'a str])>,
    ) {
        let subpaths = get_one_level_subpaths(paths);
        let expected = expected
            .into_iter()
            .map(|(p, paths)| {
                (
                    Path::new(p),
                    if paths.is_empty() {
                        None
                    } else {
                        Some(paths.iter().copied().collect::<HashSet<_>>())
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        assert_eq!(subpaths, expected);
    }

    #[rstest]
    #[case([], [], true)]
    #[case(["1.txt", ".env", ".venv/bin/test.sh"], [], false)]
    #[case([], ["1.txt", ".env", ".venv/bin/test.sh"], false)]
    #[case(
        ["1.txt", ".env", ".venv", ".venv/bin", ".venv/bin/test.sh"],
        [".venv/bin/test.sh", ".env", "1.txt", ".venv", ".venv/bin"],
        true,
    )]
    #[case(
        ["1.txt", ".env", ".envrc", ".venv", ".venv/bin", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        ["1.txt", ".env", ".envrc", ".venv", ".venv/notbin", ".venv/notbin/test.sh", ".venv/lib/b.pth", ".venv/pyvenv.cfg"],
        false,
    )]
    fn test_gen_all_subpath_digests<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] other_paths: impl IntoIterator<Item = &'a str>,
        #[case] expected: bool,
    ) {
        let subpath_digests = gen_all_subpath_digests(paths).collect::<HashMap<_, _>>();
        let other_subpath_digests = gen_all_subpath_digests(other_paths).collect::<HashMap<_, _>>();
        if expected {
            assert_eq!(subpath_digests, other_subpath_digests);
        } else {
            assert_ne!(subpath_digests, other_subpath_digests);
        }
    }
}
