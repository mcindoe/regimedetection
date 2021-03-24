# Regime Detection

Provides an implementation of the Azran-Ghahramani clustering algorithm, as detailed in the paper *A New Approach to Data-Driven Clustering* (see references, referred to as *The AG-Paper*), as well as some examples in both non-financial and financial settings, as detiled in *Our Paper* (to be named)

# Setup

* `git clone` the repository to a folder named `regimedetection`
    - `git clone https://github.com/mcindoe/regimedetection.git`
* Add the `regimedetection` folder to your `PYTHONPATH` environment variable
    - I refer to modules as e.g. `from regimedetection.src.metrics import euclidean_distance`
    - Open to suggestions on how to better accomplish this
* Create a new virtual environment and `pip install -r requirements.txt`

* The `signatory` package is tricky to get installed. For full instructions, see the [installation guide](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html). This project uses:
    * Python 3.8.8 (may be [installed from source](https://www.python.org/downloads/release/python-388/), or alternatively consider `pyenv` or the `conda` implementation). At the time of writing, `signatory` only supports Python 3.6, 3.7 or 3.8
    * `PyTorch 1.7.1` (again, only specific versions of `PyTorch` play nicely with `signatory`)
    * `signatory v1.2.4`. See the [installation guide](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html), although the following command should suffice:
        - `pip install signatory==<SIGNATORY_VERSION>.<TORCH_VERSION> --no-cache-dir --force-reinstall`
        - `pip install signatory==1.2.4.1.7.1 --no-cache-dir --force-reinstall`
        - Explanations for this command may be found in the installations instruction page
        - This must be installed after `PyTorch`, hence it is not included in the `requirements.txt`

# Coding Standards

* Currently using `Black` with a line length of 100, since I find the 79 limit too limiting

# On Empty Partitions in Multiscale-K-Prototypes

This was not mentioned in the paper by Azran & Ghahramani:

In the multiscale-k-prototypes algorithm, at each iteration the current cluster elements are used to determine the cluster centres in the next iteration. If, however, a given cluster is empty, it is not clear what to do. The solution implemented here is to do a *star-shaped-init* style solution (see Section 4.2 and Algorithm 2 of The AG-Paper). That is, from the collection of prototypes corresponding to all points in the space, we choose the prototype which has maximal KL-divergence from the already-assigned cluster centres. This is repeated until a prototype is assigned to each cluster index for the next iteration.

Note that this means that in the next iteration, the cluster with a *manually-assigned* cluster centre is guaranteed to have at least one element, since there is an element of the space with zero KL-divergence to the cluster centre.
