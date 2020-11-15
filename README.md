# Regime Detection

Code not yet ready for review. AZ-Clustering algorithm needs to be cleaned and some examples introduced. In particular some changes have been made and not checked that the script even works at present.

## TODO

1. Add rest of code in to-add.py and check for inaccuracies
2. Add examples folder and some examples of using the AZ-Clustering module
3. Bit more structure to the repo. Rename azran-ghahramani.py -> clustering.py and move utils such as closest-even-integer to a separate utils file.

## Some Open Issues

1. In K-Prototypes algorithm, what are we to do if one of the partitions is empty? E.g. if kth cluster is entry, what is new-prototypes[k]? I suppose this comes up in K-means clustering as well. Currently I'm leaving the partition unchanged.
2. Verify the claim in 7.2 of the reference. I didn't agree with this when doing the thesis, and will be a major speedup if we can get this working
