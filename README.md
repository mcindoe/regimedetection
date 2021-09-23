# Regime Detection

Provides an implementation of the Azran-Ghahramani clustering algorithm, as detailed in the paper *A New Approach to Data-Driven Clustering* (see references, referred to as *The AG-Paper*), as well as some examples in both non-financial and financial settings, as detailed in [Market Regime Classification with Signatures](https://arxiv.org/abs/2107.00066)

# Setup

## TL;DR (Condensed and Less Helpful Instructions)

* Run `install.sh` in the home directory in a new Python [virtual environment](https://www.section.io/engineering-education/introduction-to-virtual-environments-and-dependency-managers/) with Python 3.9 (or check the full instructions below to see if newer Python versions are now supported)
* Add the parent of the project's root directory to Path or `PYTHONPATH` (so that `from regimedetection.module import function` runs from any script)

## Instructions

This repository makes use of the [signatory](https://github.com/patrick-kidger/signatory) package, which must be installed after [PyTorch](https://pytorch.org/), and the version must be selected with reference to the installed PyTorch version.

At the time of writing, signatory's [installation guide](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html) informs the reader that signatory is supported for Python 3.6-3.9 and PyTorch versions 1.6.0-1.9.0. Signatory must also be installed after PyTorch. All packages other than PyTorch and signatory may be installed in any order, and later versions of these will likely not cause any issues.

The following steps may be used to set up the repository on a Linux machine. Instructions for other operating systems will be added shortly.

1. `git clone` the repository: `git clone https://github.com/mcindoe/regimedetection.git`
2. Make the installation script executable: `chmod +x install.sh`
3. Run the `install.sh` installation script to install the packages in the required order
4. Add the parent directory of this `regimedetection` repository to the `PYTHONPATH` environment variable
    - This allows imports such as e.g. `from regimedetection.src.metrics import euclidean_distance` to work from any working directory
    - In MacOS / Linux, add `export PYTHONPATH=$PYTHONPATH:/path/to/parent/dir` in your shell's config file, e.g. `~/.bashrc` if using bash, or `~/.zshrc` if using zsh.

# On Empty Partitions in Multiscale-K-Prototypes

In the multiscale-k-prototypes algorithm, at each iteration the current cluster elements are used to determine the cluster centres in the next iteration. If, however, a given cluster is empty, it is not clear what to do. The solution implemented here is to do a *star-shaped-init* style solution (see Section 4.2 and Algorithm 2 of The AG-Paper). That is, from the collection of prototypes corresponding to all points in the space, we choose the prototype which has maximal KL-divergence from the already-assigned cluster centres. This is repeated until a prototype is assigned to each cluster index for the next iteration.

Note that this means that in the next iteration, the cluster with a *manually-assigned* cluster centre is guaranteed to have at least one element, since there is an element of the space with zero KL-divergence to the cluster centre.

This approach is briefly mentioned in [our preprint](https://arxiv.org/abs/2107.00066), but it is worth being aware of.
