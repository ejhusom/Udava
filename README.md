# UDAVA

Unsupervised learning for DAta VAlidation.


## Installation


Developed using Python3.8. You can install the required modules by creating a
virtual environment and install the `requirements.txt`-file (run these commands
from the main folder):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


## Udava as a Service


Start the server by running:

```
python3 src/api.py
```



## Parameters


- `featurize`
    - `dataset`
    - `window_size`
    - `overlap`
    - `timestamp_column`
    - `columns`
- `cluster`
    - `learning_method`
    - `n_clusters`
    - `max_iter`
    - `use_predefined_centroids`
    - `fix_predefined_centroids`
    - `annotations_dir`
    - `min_segment_length`: A segment is defined as a section of the time series that has an uninterrupted sequence of data points with the same cluster label. This parameter defines the minimum length such a sequence should have. If a segment is shorter than this length, the data points will be reassigned to another cluster.
- `evaluate`
