# Numerical reconstruction for inverse problems

This repository contains implementations of numerical reconstruction algorithms
for various inverse problems. It also hosts, and will continue to collect,
Jupyter notebooks for experiments reported in the author's papers.

The repository is a collection of research code and experiments. It is not yet a
library, and its interfaces and organization may change as the work develops.

## Geodesic ray transforms (`grt`)

The [`grt`](grt/) directory contains code specifically for geodesic ray
transforms, including geodesic computation, discretization of the forward
operator, and numerical reconstruction in two and three dimensions.

- [`grt/planar`](grt/planar/): two-dimensional geometry and ray transforms.
- [`grt/spatial`](grt/spatial/): three-dimensional geometry and ray transforms.
- [`grt/reconstruction.py`](grt/reconstruction.py): steepest gradient descent
  and conjugate gradient reconstruction.
- [`common`](common/): shared utilities.

## Experiments and notebooks

The [`experiments`](experiments/) directory hosts notebooks and associated
outputs for the author's numerical experiments. Current geodesic ray transform
experiments are organized into [planar](experiments/grt/planar/) and
[spatial](experiments/grt/spatial/) cases, with spherical, Poincaré, and focusing
lens metrics.

Additional notebooks will be added as experiments from the author's papers are
made available.

## Required citation for `grt`

If you use `grt` in your research, please cite the author's paper:

> Yernat M. Assylbekov. *Inverting the geodesic ray transform with finite
> measurements: stability and reconstruction*. arXiv:2609.40015, 2026.

The paper is available on [arXiv](https://arxiv.org/abs/2609.40015), and its
manuscript source is available in [`grt_finite.tex`](grt_finite.tex).
This is a request for academic attribution, not a condition of use.
The [`CITATION.cff`](CITATION.cff) file identifies this paper as the preferred
citation for GitHub's “Cite this repository” feature.

```bibtex
@misc{assylbekov2026invertinggeodesicraytransform,
  author = {Assylbekov, Yernat M.},
  title = {Inverting the geodesic ray transform with finite measurements: stability and reconstruction},
  year = {2026},
  eprint = {2609.40015},
  archivePrefix = {arXiv},
  primaryClass = {math.AP},
  doi = {10.48550/arXiv.2609.40015},
  url = {https://arxiv.org/abs/2609.40015}
}
```
