# Regime Detection

Provides an implementation of the AG-Clustering algorithm. More examples to follow shortly.

# Setup

* `git clone` the repository to a folder named `regimedetection`
    - `git clone https://github.com/mcindoe/RegimeDetection.git`
    - `mv RegimeDetection regimedetection`
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

# Some Open Issues

1. In K-Prototypes algorithm, what are we to do if one of the partitions is empty? E.g. if kth cluster is entry, what is `new-prototypes[k]`? I suppose this comes up in K-means clustering as well. Currently I'm leaving the partition unchanged.
