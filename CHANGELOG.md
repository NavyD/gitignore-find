## v1.1.0 (2024-12-01)

### Feat

- Cache paths to improve performance

## v1.0.0 (2024-11-30)

### Feat

- Rewrite functions to improve performance

## v0.3.2 (2024-11-26)

### Fix

- python type stubs not work for find_ignoreds fn
- Required py version from 3.8 to 3.9 in pyproject.toml

## v0.3.1 (2024-11-26)

### Fix

- **ci**: Update from deprecated macos-12 to macos-13

## v0.3.0 (2024-11-26)

### Feat

- Add pre-commit
- Optimized find_ignoreds fn performance
- Add exclude_ignoreds arg feature to exclude some paths
- Upgrade deps for breaking changes

### Fix

- **ci**: failed to build pyo3

## v0.2.0 (2024-01-30)

### Feat

- Release v0.2.0
- Optimize duplicate paths

## v0.1.0 (2024-01-30)

### Feat

- **ci**: Add multi python versions build
- Refactor the find_ignoreds fn using `ignore` instead of `git2` lib
- Init
