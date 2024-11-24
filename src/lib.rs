use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Error, Result};
use globset::{GlobBuilder, GlobSetBuilder};
use ignore::gitignore::Gitignore;
use itertools::Itertools;
use jwalk::{rayon::prelude::*, WalkDir};
use log::{debug, log_enabled, trace, warn};
use pyo3::{
    prelude::*,
    types::{PySequence, PyString},
};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

/// A Python module implemented in Rust.
#[pymodule]
fn gitignore_find(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: 等待兼容pyo3 0.23版本 https://github.com/vorner/pyo3-log
    // pyo3_log::init();
    m.add_function(wrap_pyfunction!(find_ignoreds, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (paths, excludes=None, exclude_ignoreds=None))]
fn find_ignoreds(
    paths: &Bound<'_, PyAny>,
    excludes: Option<&Bound<'_, PySequence>>,
    exclude_ignoreds: Option<&Bound<'_, PySequence>>,
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
    let exclude_ignoreds = exclude_ignoreds
        .map(|e| e.extract::<Vec<String>>())
        .unwrap_or_else(|| Ok(vec![]))?;
    debug!(
        "Finding git ignored paths with exclude globs \
        {:?} and exclude ignored globs {:?} in paths: {:?}",
        excludes, exclude_ignoreds, paths
    );
    let ignoreds = find(paths, excludes, exclude_ignoreds)?;
    debug!("Found {} ignored paths: {:?}", ignoreds.len(), ignoreds);
    Ok(ignoreds)
}

pub fn find(
    paths: impl IntoIterator<Item: AsRef<Path> + Clone + Debug>,
    excludes: impl IntoIterator<Item: AsRef<str> + Clone + Debug>,
    exclude_ignoreds: impl IntoIterator<Item: AsRef<str> + Clone + Debug>,
) -> Result<Vec<PathBuf>> {
    let (paths, excludes, exclude_ignoreds) = (
        paths.into_iter().collect_vec(),
        excludes.into_iter().collect_vec(),
        exclude_ignoreds.into_iter().collect_vec(),
    );
    if log_enabled!(log::Level::Debug) {
        debug!(
            "Finding git ignored paths with exclude globs \
            {:?} and exclude ignored globs {:?} in {} paths: {:?}",
            excludes,
            exclude_ignoreds,
            paths.len(),
            paths
        );
    }
    let all_paths = find_all_paths(&paths, excludes)?;

    debug!(
        "Finding git ignored paths with exclude patterns {:?} in all {} paths",
        exclude_ignoreds,
        all_paths.len()
    );
    let ignoreds = find_gitignoreds(
        all_paths.into_iter().map(Arc::new),
        GlobPathPattern::new(exclude_ignoreds)?,
    )
    // SAFETY: only one ref
    .map(|p| Arc::try_unwrap(p).unwrap())
    .collect::<Vec<PathBuf>>();
    debug!("Found {} ignored paths: {:?}", ignoreds.len(), ignoreds);
    Ok(ignoreds)
}

/// 找到指定路径中的所有文件与目录
fn find_all_paths<P, Q>(
    paths: impl IntoIterator<Item = P>,
    excludes: impl IntoIterator<Item = Q>,
) -> Result<Vec<PathBuf>>
where
    P: AsRef<Path>,
    Q: AsRef<str>,
{
    let exclude_pat = excludes
        .into_iter()
        .try_fold(GlobSetBuilder::new(), |mut gs, s| {
            let glob = GlobBuilder::new(s.as_ref())
                .literal_separator(true)
                .build()?;
            gs.add(glob);
            Ok::<_, Error>(gs)
        })
        .and_then(|b| b.build().map_err(Into::into))?;

    let exclude_pat = Arc::new(exclude_pat);
    paths
        .into_iter()
        .flat_map(|path| {
            let exclude_pat = exclude_pat.clone();
            let path = path.as_ref();
            debug!("Traversing paths in directory {}", path.display());
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
        })
        .collect::<Result<Vec<PathBuf>>>()
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

/// 对于paths中`.gitignore`存在的目录，返回所有被其忽略的文件和目录
///
/// ## 移除
///
/// 不能先对paths进行移除，因为后续merge依赖paths路径的完整性，如果移除了paths部分路径文件，
/// 会导致后续合并时认为目录仍然完整，从而合并了不应该合并的目录
///
/// 由于gi只匹配文件夹而不会对其中的文件匹配如`.venv`规则不会匹配`.venv/bin/test.sh`
/// 会导致忽略的路径不包含子路径。一个简单的实现，在检查.gitignore时，
/// 由于cur_ignoreds不包含子路径，在排除路径时需要获取所有路径，
/// 再从ignoreds中移除被排除的路径与父路径，这个路径会在后面被合并不会导致完整性问题，但存在性能问题，
/// 一旦cur_ignoreds包含的所有路径稍大会消耗的时间指数级上升，不可取
fn find_gitignoreds<I, T, P, E>(paths: I, exclude_pat: T) -> impl Iterator<Item = E>
where
    I: IntoIterator<Item = E>,
    E: Deref<Target = P> + Clone + Debug + Send + Sync + Eq + Hash + Ord,
    P: AsRef<Path>,
    T: PathPattern + Send + Sync,
{
    let paths = paths.into_iter().collect::<Vec<_>>();
    let merge_paths = MergePaths::new(paths.iter().map(|p| p.as_ref()));
    let ignoreds = paths
        .par_iter()
        .filter_map(|path| {
            let path: &Path = path.as_ref();
            // 跳过非gitignore的目录
            if !path.ends_with(".gitignore") || !path.is_file() {
                return None;
            }

            trace!("Loading gitignore rule from {}", path.display());
            let (gi, err) = Gitignore::new(path);
            if let Some(e) = err {
                warn!(
                    "Ignore loaded gitignore rule error in {}: {}",
                    path.display(),
                    e
                );
            }

            #[cfg(debug_assertions)]
            {
                let txt = match std::fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(e) => format!("Failed to reading path {} by error: {}", path.display(), e),
                };
                trace!(
                    "Loaded gitignore {} rules for gitignore content: {}",
                    gi.len(),
                    txt.replace("\r\n", "\\r\\n").replace("\n", "\\n")
                );
            }

            let dir: &Path = path.parent()?;
            // 获取除了 被排除 的所有路径
            let cur_ignoreds = paths
                .par_iter()
                .filter(|path| {
                    let p = path.as_ref();
                    p != dir
                        && p.starts_with(dir)
                        && (
                            // gi只匹配文件夹而不会对其中的文件匹配如`.venv`规则不会匹配`.venv/bin/test.sh`会导致忽略的路径不包含子路径
                            // 添加新条件以匹配子路径
                            gi.matched(p, p.is_dir()).is_ignore()
                                || p.ancestors()
                                    .skip(1)
                                    // pp必然是dir
                                    .any(|pp| gi.matched(pp, true).is_ignore())
                        )
                })
                // 过滤不需要忽略的路径
                .filter(|p| exclude_pat.is_empty() || !exclude_pat.is_match(p.as_ref()))
                .collect::<Vec<_>>();
            #[cfg(debug_assertions)]
            trace!(
                "Found {} ignoreds paths in directory {}: {:?}",
                cur_ignoreds.len(),
                dir.display(),
                cur_ignoreds,
            );
            Some(cur_ignoreds)
        })
        .flatten()
        .cloned()
        .collect::<HashSet<_>>();
    debug!(
        "Mergeing {} ignored paths in {} paths",
        ignoreds.len(),
        paths.len()
    );
    merge_paths.merge_rc_owned(ignoreds)
}

struct MergePaths<'a> {
    all_subpath_digests: HashMap<&'a Path, Option<[u8; 32]>>,
}

impl<'p> MergePaths<'p> {
    pub fn new<P>(paths: impl IntoIterator<Item = &'p P>) -> Self
    where
        P: AsRef<Path> + 'p + ?Sized + Ord + Sync + Send,
    {
        Self {
            all_subpath_digests: Self::gen_subpath_digests(paths),
        }
    }

    /// 生成所有路径对应的 递归所有子路径 的digests。
    ///
    /// 如果一个路径没有子路径则`digest=None`
    ///
    /// ## 实现
    ///
    /// 排序后自低向上查找每个路径对应的子路径。对于每个路径，使用子路径的digest计算digest，
    /// 不会真的检查递归的子路径，算法复杂度是O(N)
    fn gen_subpath_digests<'a, P>(
        paths: impl IntoIterator<Item = &'a P>,
    ) -> HashMap<&'a Path, Option<[u8; 32]>>
    where
        P: AsRef<Path> + 'a + ?Sized,
    {
        fn path_sort_desc_key(p: &Path) -> impl Ord + Send + '_ {
            (
                std::cmp::Reverse(p.ancestors().count()),
                p.file_name().and_then(|s| s.to_str()),
            )
        }

        let paths = {
            let mut v = paths.into_iter().map(|p| p.as_ref()).collect_vec();
            // 加速合并
            if v.len() > 100000 {
                v.par_sort_by_cached_key(|p| path_sort_desc_key(p));
            } else {
                v.sort_by_cached_key(|p| path_sort_desc_key(p));
            }
            v
        };

        #[cfg(debug_assertions)]
        trace!(
            "Getting one tier subpaths in {} paths: {:?}",
            paths.len(),
            paths
        );
        // 保存路径对应的子路径  如果一个路径没有子路径则为空
        let one_tier_subpaths = paths.iter().copied().fold(
            HashMap::new(),
            |mut path_subpaths: HashMap<&Path, SmallVec<[&Path; 4]>>, p| {
                path_subpaths.entry(p).or_default();
                if let Some(pp) = p.parent() {
                    // 相对路径的顶级路径为 空 时，保留Key即可 可能为空 或 保留之前存在了子路径
                    if !pp.to_string_lossy().is_empty() {
                        path_subpaths.entry(pp).or_default().push(p);
                    }
                }
                path_subpaths
            },
        );
        debug!("Generating all subpath digests for {} paths", paths.len());
        type DigestTy = Sha256;
        let subpath_digests = paths.iter().copied().fold(
            HashMap::<&Path, Option<[u8; 32]>>::new(),
            |mut subpath_digests, path| {
                // SAFETY: paths的所有键都存在于one_tier_subpaths中
                let subpaths = one_tier_subpaths
                    .get(path)
                    .unwrap_or_else(|| panic!("Not found subpaths for path {}", path.display()));
                #[cfg(debug_assertions)]
                trace!(
                    "Getting digest from {} next subpaths={:?} in current path {}",
                    subpaths.len(),
                    subpaths,
                    path.display()
                );
                // 当前路径无子路径
                let p_digest = if subpaths.is_empty() {
                    None
                } else {
                    // 当前存在子路径
                    let p_digest: [u8; 32] = subpaths
                        .iter()
                        .fold(DigestTy::new(), |p_digest, subp| {
                            // SAFETY: 降序保证了子路径先于父路径计算digest，所以当前路径的子路径一定是计算过的 或 无子路径
                            if let Some(subp_digest) =
                                subpath_digests.get(subp).unwrap_or_else(|| {
                                    panic!(
                                        "Not found digest for sub path {} of path {}",
                                        subp.display(),
                                        path.display()
                                    )
                                })
                            {
                                p_digest.chain_update(subp_digest)
                            } else {
                                // 当前路径不存在子路径digest时如 `/a/b/1.txt`中的b子路径`1.txt`无子路径即不会
                                // 有digest，在计算到b路径的subpaths时为空，此时使用1.txt路径计算
                                let d: [u8; 32] = DigestTy::new()
                                    .chain_update(subp.display().to_string())
                                    .finalize()
                                    .into();
                                p_digest.chain_update(d)
                            }
                        })
                        .finalize()
                        .into();
                    Some(p_digest)
                };
                subpath_digests.insert(path, p_digest);
                subpath_digests
            },
        );
        subpath_digests
    }

    #[allow(dead_code)]
    pub fn merge<'b, P>(
        &self,
        paths: impl IntoIterator<Item = &'b P>,
        // ) -> Vec<&'b P>
    ) -> impl Iterator<Item = &'b P>
    where
        P: AsRef<Path> + Debug + 'b + ?Sized + Sync + Send + Ord + Hash,
    {
        self.merge_rc_owned(paths.into_iter().map(Arc::new))
            // SAFETY: p only one ref
            .map(|p| Arc::try_unwrap(p).unwrap())
        // todo!()
    }

    /// 检查当前的paths与自身的差异后合并所有可能的路径并返回。
    ///
    /// 通过比较当前路径是否被包含在其父路径中 且 其所有子路径都存在于整体中时，
    /// 将合并这个路径
    pub fn merge_rc_owned<I, E, P>(&self, paths: I) -> impl Iterator<Item = E>
    where
        I: IntoIterator<Item = E>,
        E: Deref<Target = P> + Debug + Clone + Sync + Send + Ord + Hash,
        P: AsRef<Path>,
    {
        let paths = paths.into_iter().collect::<Vec<_>>();
        debug!("Generating subpath digests for {} paths", paths.len());
        let cur_subpath_digests = Self::gen_subpath_digests(paths.iter().map(|p| p.as_ref()));

        debug!(
            "Merging {} paths for all {} paths",
            paths.len(),
            self.all_subpath_digests.len(),
        );
        paths
            .par_iter()
            .filter(|&p| {
                let p = p.as_ref();
                // SAFETY: p always in cur_subpaths
                let p_subpath_digest = cur_subpath_digests
                    .get(p)
                    .unwrap_or_else(|| panic!("Not found digest for current path {}", p.display()));

                let mut p_ancestors = p
                    .ancestors()
                    // 排除第1个 self
                    .skip(1)
                    // 排除`Path::new("a/b")`时最后一个为空
                    .filter(|pp| !pp.as_os_str().is_empty())
                    .peekable();

                // 这个p是顶级路径如相对路径`a`或绝对路径`/`
                let keep = if p_ancestors.peek().is_none() {
                    // 当前路径 无子路径
                    // 或 子路径全部都存在于整体中时 保留这个路径
                    // 表示是需要合并的路径 返回 true
                    p_subpath_digest.is_none()
                        || self.all_subpath_digests.get(p) == Some(p_subpath_digest)
                } else {
                    // 当前路径 不存在子路径 或 所有子路径与整体一致，
                    // 且 当前路径的任意一个父路径 已存在 且 这个父路径所有的子路径都在整体中，
                    // 那么表示可合并 这个路径即可忽略 返回false
                    (p_subpath_digest.is_none()
                        || self.all_subpath_digests.get(p) == Some(p_subpath_digest))
                        && !p_ancestors.any(|pp| {
                            cur_subpath_digests.contains_key(pp)
                                && cur_subpath_digests.get(pp) == self.all_subpath_digests.get(pp)
                        })
                };

                #[cfg(debug_assertions)]
                trace!("Path {} is keep={}", p.display(), keep);
                keep
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

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
    fn mock_git_paths<P, Q>(
        gitignore_items: impl IntoIterator<Item = P>,
        paths: impl IntoIterator<Item = Q>,
        repo_dir: impl AsRef<Path>,
    ) -> impl Iterator<Item = PathBuf>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        let repo_dir = repo_dir.as_ref();

        // write .gitignore file
        let gitignore_items = gitignore_items.into_iter().collect_vec();
        if !gitignore_items.is_empty() {
            let gitignore_path = repo_dir.join(".gitignore");
            std::fs::write(
                &gitignore_path,
                gitignore_items
                    .iter()
                    .map(|i| i.as_ref().display())
                    .join("\n"),
            )
            .unwrap();
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
    #[test]
    #[case::all_empty([], [], [])]
    #[case::no_excludes(["1.txt", ".env", ".envrc"], [], ["1.txt", ".env", ".envrc"])]
    #[case::excludes(["1.txt", ".env", ".envrc"], ["**/.env*"], ["1.txt"])]
    fn test_find_paths<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] excludes: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let tmpdir = tempfile::tempdir().unwrap();
        let paths = paths.into_iter().collect_vec();
        // NOTE: tmpdir非引用会导致提前移除目录
        let _git_paths = mock_git_paths([] as [&str; 0], &paths, &tmpdir).collect_vec();

        let mut all_paths = find_all_paths([&tmpdir], excludes)?;
        // 包含根目录tmpdir
        assert!(all_paths.contains(&tmpdir.path().to_path_buf()));
        all_paths.retain(|p| p.as_path() != tmpdir.path());
        all_paths.sort();

        let expected: Vec<_> = expected
            .into_iter()
            .map(|s| tmpdir.path().join(s))
            .sorted()
            .collect();
        assert_eq!(all_paths, expected);
        Ok(())
    }

    #[rstest]
    #[case::all_empty([], [], [], [])]
    #[case::no_gitignore_and_no_excludes(["1.txt", ".env", ".envrc"], [], [], [])]
    #[case::no_excludes(["1.txt", ".env", ".envrc"], [".env*"], [], [".env", ".envrc"])]
    #[case::no_gitignore(["1.txt", ".env", ".envrc"], [], ["**/.env"], [])]
    #[case::exclude_env(["1.txt", ".env", ".envrc"], [".env*"], ["**/.env"], [".envrc"])]
    #[case::nest_excludes(["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth"], [".env*", ".venv"], ["**/.env", "**/.venv/**", "**/.venv"], [".envrc"])]
    #[case::nest_excludes(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        [".env*", ".venv"],
        ["**/.env", "**/.venv/bin", "**/.venv/bin/**"],
        [".envrc", ".venv/lib", ".venv/pyvenv.cfg"],
    )]
    #[case::nest_excludes_without_globself(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        [".env*", ".venv"],
        ["**/.env", "**/.venv/bin/**"],
        [".envrc", ".venv/bin", ".venv/lib", ".venv/pyvenv.cfg"],
    )]
    fn test_find_gitignoreds<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] gitignore_items: impl IntoIterator<Item = &'a str>,
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
        let ignoreds = find_gitignoreds(
            git_paths.into_iter().map(Arc::new),
            GlobPathPattern::new(excludes)?,
        )
        .sorted()
        .map(|p| Arc::try_unwrap(p).unwrap())
        .collect::<Vec<PathBuf>>();
        let expected = expected
            .into_iter()
            .map(|s| tmpdir.path().join(s))
            .sorted()
            .collect::<Vec<_>>();
        assert_eq!(ignoreds, expected);
        Ok(())
    }

    static MERGE_ALL_PATHS: &[&str] = &[
        "a",
        "a/1.txt",
        "b",
        "b/.env",
        "b/.envrc",
        "c",
        // relative paths
        ".venv",
        ".venv/pyvenv.cfg",
        ".venv/bin",
        ".venv/bin/test.sh",
        ".venv/bin/activate",
        ".venv/lib",
        ".venv/lib/a.pth",
        // absolute venv paths
        "/tmp/rs/.venv",
        "/tmp/rs/.venv/pyvenv.cfg",
        "/tmp/rs/.venv/bin",
        "/tmp/rs/.venv/bin/test.sh",
        "/tmp/rs/.venv/bin/activate",
        "/tmp/rs/.venv/lib",
        "/tmp/rs/.venv/lib/a.pth",
    ];

    #[rstest]
    #[case(MERGE_ALL_PATHS, ["b", "b/.env", "b/.envrc"], ["b"])]
    #[case(MERGE_ALL_PATHS, ["b", "b/.envrc"], ["b/.envrc"])]
    #[case(MERGE_ALL_PATHS, ["c"], ["c"])]
    #[case(
        MERGE_ALL_PATHS,
        ["b", "b/.env", "c", ".venv/bin/test.sh", ".venv/pyvenv.cfg", ".venv/lib", ".venv/lib/a.pth",],
        ["b/.env", "c", ".venv/lib", ".venv/bin/test.sh", ".venv/pyvenv.cfg"]
    )]
    #[case(
        MERGE_ALL_PATHS,
        [".venv", ".envrc", ".venv/pyvenv.cfg", ".venv/lib", ".venv/lib/a.pth"],
        [".envrc", ".venv/lib", ".venv/pyvenv.cfg"],
    )]
    #[case(
        MERGE_ALL_PATHS,
        ["/tmp/rs/.venv", "/tmp/rs/.envrc", "/tmp/rs/.venv/pyvenv.cfg", "/tmp/rs/.venv/lib", "/tmp/rs/.venv/lib/a.pth"],
        ["/tmp/rs/.envrc", "/tmp/rs/.venv/lib", "/tmp/rs/.venv/pyvenv.cfg"],
    )]
    fn test_merge_paths<'a>(
        #[case] all_paths: impl IntoIterator<Item = &'a (impl AsRef<Path> + 'a + Ord + Send + Sync)>,
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = &'a str>,
        // #[case] paths: impl IntoIterator<Item = &'a (impl AsRef<Path> + 'a + Debug + Ord)>,
        // #[case] expected: impl IntoIterator<Item = &'a (impl AsRef<Path> + 'a + Ord)>,
    ) -> Result<()> {
        let paths = paths.into_iter().collect_vec();
        let all_paths = all_paths.into_iter().collect_vec();
        let mergeds = MergePaths::new(all_paths)
            .merge(paths)
            .sorted()
            .collect::<Vec<_>>();
        let expected = expected.into_iter().sorted().collect_vec();
        assert_eq!(mergeds, expected);
        Ok(())
    }

    #[rstest]
    #[case([], [], true)]
    #[case(["1.txt", ".env", ".venv/bin/test.sh"], [], false)]
    #[case([], ["1.txt", ".env", ".venv/bin/test.sh"], false)]
    #[case(["1.txt", ".env", ".venv/bin/test.sh"], [".venv/bin/test.sh", ".env", "1.txt"], true)]
    #[case(
        ["1.txt", ".env", ".envrc", ".venv/bin/test.sh", ".venv/lib/a.pth", ".venv/pyvenv.cfg"],
        ["1.txt", ".env", ".envrc", ".venv/notbin/test.sh", ".venv/lib/b.pth", ".venv/pyvenv.cfg"],
        false,
    )]
    fn test_gen_subpath_digests<'a>(
        #[case] paths: impl IntoIterator<Item = &'a str>,
        #[case] other_paths: impl IntoIterator<Item = &'a str>,
        #[case] expected: bool,
    ) {
        let subpath_digests = MergePaths::gen_subpath_digests(paths);
        let other_subpath_digests = MergePaths::gen_subpath_digests(other_paths);
        if expected {
            assert_eq!(subpath_digests, other_subpath_digests);
        } else {
            assert_ne!(subpath_digests, other_subpath_digests);
        }
    }

    #[ignore = "实机测试环境可能不一致"]
    #[rstest]
    #[case(
        ["**/.git", "**/.cargo", "**/.vscode*"],
        [] as [&str; 0],
        [
            "./.venv",
            "./flamegraph.svg",
            "./perf.data",
            "./target",
            "./perf.data.old",
        ]
    )]
    #[case(
        ["**/.git", "**/.cargo", "**/.vscode*"],
        ["**/target", "**/target/**"],
        [
            "./.venv",
            "./flamegraph.svg",
            "./perf.data",
            "./perf.data.old",
        ]
    )]
    fn test_integration_find_ignoreds_in_current_repo<'a>(
        #[case] excludes: impl IntoIterator<Item = &'a str>,
        #[case] exclude_ignoreds: impl IntoIterator<Item = &'a str>,
        #[case] expected: impl IntoIterator<Item = &'a str>,
    ) {
        fn path_sort_asc_fn(a: impl AsRef<Path>, b: impl AsRef<Path>) -> std::cmp::Ordering {
            let (a, b) = (a.as_ref(), b.as_ref());
            a.ancestors()
                .count()
                .cmp(&b.ancestors().count())
                .then_with(|| a.cmp(b))
        }
        let mut ignoreds = find(["."], excludes, exclude_ignoreds).unwrap();
        // ignoreds.sort_unstable_by_key(|p| (std::cmp::Reverse(p.ancestors().count()), p));
        ignoreds.sort_unstable_by(|a, b| path_sort_asc_fn(a, b));
        let expected = expected
            .into_iter()
            .map(PathBuf::from)
            .sorted_unstable_by(|a, b| path_sort_asc_fn(a, b))
            .collect::<Vec<PathBuf>>();
        assert_eq!(ignoreds, expected);
    }
}
