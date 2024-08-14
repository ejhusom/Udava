# UDAVA

Unsupervised learning for DAta VAlidation.


## Udava as a Service

Choose option A or B to start the Udava service on your computer.

### A) Run Udava directly on host

You can install the required modules by creating a
virtual environment and install the `requirements.txt`-file (run these commands
from the main folder):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Start the server by running:

```
python3 src/api.py
```


### B) Running Udava in Docker


```
docker build -t udava -f Dockerfile .
docker run -p 5000:5000 -it -v $(pwd)/assets:/usr/Udava/assets -v $(pwd)/.dvc:/usr/Udava/.dvc udava
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
