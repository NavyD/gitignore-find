from collections.abc import Sequence
import os
import pathlib

_PATH = str | os.PathLike[str] | pathlib.Path

def find_ignoreds(
    paths: _PATH | Sequence[_PATH], excludes: Sequence[str] | None = None
) -> list[str]:
    """查找指定paths下所有git仓库中被忽略的文件和目录

    Args:
        paths (_PATH | Sequence[_PATH]): 需要查找的路径
        excludes (Sequence[str] | None, optional): 在.gitignore忽略的路径中需要排除的路径globs。
            如`.gitignore`中存在规则`.env`忽略`.env`文件，如果不指定排除规则会返回包含这个路径，如果指定
            排除规则如`**/.env`会从返回的所有路径中排除所有包含这个路径的路径。
            Defaults to None.

    Returns:
        list[str]: 返回指定路径下被忽略的路径列表
    """
    ...
