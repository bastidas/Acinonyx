# traj_dist

Pure Python implementation of trajectory distance metrics for 2D trajectories.

## Attribution

This module is based on the **traj-dist** package by Brendan Guillouet:
- **Repository**: https://github.com/bguillouet/traj-dist
- **License**: MIT
- **Author**: Brendan Guillouet

This is a pure Python port using the `pydist` implementations instead of the Cython (`cydist`) implementations from the original package. The code has been adapted to work without requiring Cython compilation.

## Available Metrics

The following distance metrics are available:

1. **SSPD** (Symmetric Segment-Path Distance)
2. **OWD** (One-Way Distance) / **SOWD** (Symmetric One-Way Distance)
3. **Hausdorff**
4. **Frechet**
5. **Discrete Frechet**
6. **DTW** (Dynamic Time Warping)
7. **LCSS** (Longest Common Subsequence)
8. **ERP** (Edit distance with Real Penalty)
9. **EDR** (Edit Distance on Real sequence)

All distances (except Discrete Frechet and Frechet) are available with both **Euclidean** and **Spherical** (Haversine) distance options.

## Usage

```python
import traj_dist.distance as tdist
import numpy as np

# Create trajectories (n x 2 numpy arrays)
traj_A = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
traj_B = np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])

# Simple distance
dist = tdist.sspd(traj_A, traj_B)
print(dist)

# Pairwise distance
traj_list = [traj_A, traj_B, ...]
pdist = tdist.pdist(traj_list, metric="sspd")
print(pdist)

# Distance between two lists of trajectories
cdist = tdist.cdist(traj_list, traj_list, metric="sspd")
print(cdist)
```

## Dependencies

- numpy
- geohash2==1.1
- shapely>=1.6.4

## References

See the original repository for academic references and citations:
https://github.com/bguillouet/traj-dist
