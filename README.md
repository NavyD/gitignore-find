# gitignore-find

查找指定目录下所有被`.gitignore`文件忽略的路径，功能与`git check-ignore **/*`类似：

* 允许指定多个目录并检查其中所有的`.gitignore`文件
* 被`.gitignore`文件忽略的路径会尝试合并避免路径过多
* 超级快！

常见的用法是找出home目录下所有git仓库下忽略的目录用于从备份目录中排除

## 安装

目前只提供python扩展，使用pip从pypi安装

```sh
pip install gitignore-find
```

>提供了一个简单的命令行程序find但只能使用源码构建`cargo build --example find -r`

### 运行

```python
import gitignore_find
import logging

logging.basicConfig(level=5)
# logging.basicConfig(level=logging.DEBUG)

ignoreds = gitignore_find.find_ignoreds(
    ["."],
    excludes=["**/.git/**", "**/.cargo", "**/.vscode*"],
    exclude_ignoreds=["**/.venv/bin/**", "**/.env"],
)
print("\n".join(ignoreds))
```

## 性能

使用`cargo bench`测试当前目录22196个路径仅需要200ms

```console
$ cargo bench
    Finished `bench` profile [optimized + debuginfo] target(s) in 0.10s
     Running unittests src/lib.rs (target/release/deps/gitignore_find-c0f4f12deaf970af)
# ...
     Running benches/bench_find.rs (target/release/deps/bench_find-79a4c8a597bc913b)
Gnuplot not found, using plotters backend
gitignore find          time:   [208.50 ms 211.79 ms 215.37 ms]
                        change: [-2.6305% -0.7100% +1.1692%] (p = 0.50 > 0.05)
                        No change in performance detected.
```

在6核9750H SSD设备 wsl debian中运行下面是测试示例从home目录60万个路径中的1024个`.gitignore`文件找出忽略路径的用时是21秒左右

```console
$ time cargo run --example find -r -- -i '**/.env*' -e '**/.vscode/**' ~
    Finished `release` profile [optimized] target(s) in 0.09s
     Running `target/release/examples/find -i '**/.env*' -e '**/.vscode/**' /home/navyd`
[2024-11-25T10:20:54.037Z DEBUG gitignore_find] Finding git ignored paths with exclude globs ["**/.vscode/**"] and exclude ignored globs ["**/.env*"] in 1 paths: ["/home/navyd"]
[2024-11-25T10:20:54.037Z DEBUG gitignore_find] Traversing paths in directory /home/navyd
[2024-11-25T10:20:54.483Z DEBUG gitignore_find] Finding git ignored paths with exclude patterns ["**/.env*"] in all 603140 paths
[2024-11-25T10:20:56.514Z DEBUG gitignore_find] Finding .gitignore files in 603140 paths
[2024-11-25T10:20:56.853Z DEBUG gitignore_find] Finding ignored paths with 1024 gitignores and exclude pattern GlobPathPattern { patterns: ["**/.env*"] } from all 603140 paths
[2024-11-25T10:21:14.117Z DEBUG gitignore_find] Mergeing 116753 ignored paths in 603140 paths
[2024-11-25T10:21:14.117Z DEBUG gitignore_find] Generating subpath digests for 116753 paths
[2024-11-25T10:21:14.559Z DEBUG gitignore_find] Merging 116753 paths for all 603140 paths
[2024-11-25T10:21:14.596Z DEBUG gitignore_find] Found 884 ignored paths for ["/home/navyd"]
# ...
cargo run --example find -r -- -i '**/.env*' -e '**/.vscode/**' ~  210.91s user 3.57s system 1035% cpu 20.712 total
```
