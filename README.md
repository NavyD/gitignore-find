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
    excludes=["**/.git/**", "**/.cargo", "**/.vscode*", "**/.env"],
)

print("\n".join(ignoreds))
```

## 性能

在6核9750H SSD设备 wsl debian中运行下面是测试示例从home目录60万个路径中的1061个`.gitignore`文件找出忽略路径的用时是5秒左右，如果有缓存可以减少到2秒不到

```console
$ hyperfine --warmup 3 'target/release/examples/find ~'
Benchmark 1: target/release/examples/find ~
  Time (mean ± σ):      1.813 s ±  0.072 s    [User: 9.317 s, System: 3.497 s]
  Range (min … max):    1.743 s …  1.945 s    10 runs

$ hyperfine --prepare 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches' 'target/release/examples/find ~'
Benchmark 1: target/release/examples/find ~
  Time (mean ± σ):      5.167 s ±  0.179 s    [User: 12.203 s, System: 11.762 s]
  Range (min … max):    4.875 s …  5.557 s    10 runs

$ echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; time target/release/examples/find ~ >/dev/null
[2024-12-01T04:46:54.270Z DEBUG gitignore_find] Finding git ignored paths with exclude globs [] in 1 paths: ["/home/navyd"]
[2024-12-01T04:46:54.270Z DEBUG gitignore_find] Finding all paths in /home/navyd
[2024-12-01T04:46:54.270Z TRACE gitignore_find] Traversing paths in directory /home/navyd
[2024-12-01T04:46:57.577Z TRACE gitignore_find] Found 611706 paths for /home/navyd
[2024-12-01T04:46:57.984Z DEBUG gitignore_find] Finding ignored paths with 1061 gitignores and exclude pattern GlobPathPattern { patterns: [] } in /home/navyd
[2024-12-01T04:46:58.831Z DEBUG gitignore_find] Found 120054 ignored paths for all paths "/home/navyd"
[2024-12-01T04:46:58.851Z TRACE gitignore_find] Getting sub paths from 120054 ignoreds paths
[2024-12-01T04:46:59.241Z TRACE gitignore_find] Traversing all sub paths of 1061 .gitignore paths
[2024-12-01T04:46:59.252Z DEBUG gitignore_find] Merging 120054 ignored paths
[2024-12-01T04:46:59.330Z DEBUG gitignore_find] Found 984 ignored paths for ["/home/navyd"]
target/release/examples/find ~ > /dev/null   11.92s  user 11.44s system 460% cpu 5.074 total
avg shared (code):         0 KB
avg unshared (data/stack): 0 KB
total (sum):               0 KB
max memory:                207 MB
page faults from disk:     33
other page faults:         2994
```
