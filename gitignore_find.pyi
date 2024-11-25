from collections.abc import Sequence
import os
import pathlib

_PATH = str | os.PathLike[str] | pathlib.Path

def find_ignoreds(
    paths: _PATH | Sequence[_PATH],
    excludes: Sequence[str] | None = None,
    exclude_ignoreds: Sequence[str] | None = None,
) -> list[str]:
    """查找指定paths下所有git仓库中被忽略的文件和目录

    Args:
        paths (_PATH | Sequence[_PATH]): 需要查找的路径
        excludes (Sequence[str] | None, optional): 查找路径时需要排除的路径glob. Defaults to None.
        exclude_ignoreds (Sequence[str] | None, optional): 获取忽略路径时需要排除的glob. Defaults to None.

    Returns:
        list[str]: 返回指定路径下被忽略的路径列表
    """
    ...
