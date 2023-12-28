## Setup Evaluation Datasets
Below are the datasets used in our evaluations, along with links for downloading:
* [RoadAnomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/) 
* [Fishyscapes - Static & Lost&Found](https://fishyscapes.com/) 
* [SMIYC - RoadAnomaly21](https://uni-wuppertal.sciebo.de/s/TVR7VxukVrV7fUH/download)
* [SMIYC - RoadObstacle21](https://uni-wuppertal.sciebo.de/s/wQQq2saipS339QA/download)

Alternatively, for ease of use and standardization, you may prefer to download a preprocessed version of the datasets from [Synboost](https://github.com/giandbt/synboost), 
which includes RoadAnomaly, FS Static, and FS Lost&Found:
* [Preprocessed Datasets](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)

After downloading the dataset(s), you will need to configure the data directory in [data_set.py](../lib/dataset/data_set.py) to align with the structure of the downloaded files.
The expected data structure for these datasets in our code is as follows:

```
fishyscapes/
├── LostAndFound
│   ├── labels
│   ├── original
│   ├── semantic
└── Static
    ├── labels
    ├── original
    └── semantic

road_anomaly/
├── labels
├── original
├── semantic
└── ...
```
To quantitatively investigate the impact of domain shift on existing OOD detection methods, 
we modify the original public Fishyscapes Static dataset with post-load transformations,
including random smog, color shifting, and Gaussian blur.  You can activate these conditions by setting '--dataset FS_Static_C' in the command, as detailed in [README.md](../README.md).
