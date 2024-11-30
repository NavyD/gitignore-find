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

在6核9750H SSD设备 wsl debian中运行下面是测试示例从home目录13万个路径中的1024个`.gitignore`文件找出忽略路径的用时是5秒不到

```console
$ time cargo run -r --example find -- -e '**/.env' ~ >/dev/null
    Finished `release` profile [optimized] target(s) in 0.11s
     Running `target/release/examples/find -e '**/.env' /home/navyd`
[2024-11-30T07:19:12.740Z DEBUG gitignore_find] Finding git ignored paths with exclude globs ["**/.env"] in 1 paths: ["/home/navyd"]
[2024-11-30T07:19:12.741Z DEBUG gitignore_find] Finding .gitignore files in /home/navyd path
[2024-11-30T07:19:12.741Z TRACE gitignore_find] Traversing paths in directory /home/navyd
[2024-11-30T07:19:13.301Z DEBUG gitignore_find] Finding ignored paths with 1041 gitignores and exclude pattern GlobPathPattern { patterns: ["**/.env"] } in /home/navyd
[2024-11-30T07:19:13.301Z TRACE gitignore_find] Traversing paths in directory /home/navyd
[2024-11-30T07:19:16.582Z DEBUG gitignore_find] Found 138947 ignored paths for all paths "/home/navyd"
[2024-11-30T07:19:16.582Z TRACE gitignore_find] Excluding 138947 paths using glob pattern: GlobPathPattern { patterns: ["**/.env"] }
[2024-11-30T07:19:16.926Z TRACE gitignore_find] Getting sub paths from 138943 ignoreds paths
[2024-11-30T07:19:17.178Z TRACE gitignore_find] Traversing all sub paths of 1041 .gitignore paths
[2024-11-30T07:19:17.186Z DEBUG gitignore_find] Merging 138943 ignored paths
[2024-11-30T07:19:17.294Z DEBUG gitignore_find] Found 969 ignored paths for ["/home/navyd"]
cargo run -r --example find -- -e '**/.env' ~ > /dev/null   7.40s  user 4.80s system 257% cpu 4.730 total
```
