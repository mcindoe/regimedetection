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

# Some Open Issues

1. In K-Prototypes algorithm, what are we to do if one of the partitions is empty? E.g. if kth cluster is entry, what is `new-prototypes[k]`? I suppose this comes up in K-means clustering as well. Currently I'm leaving the partition unchanged.
