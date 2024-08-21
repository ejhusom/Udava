# UDAVA

Unsupervised learning for DAta VAlidation.


## Installation and setup

Choose option A or B to start the Udava service on your computer.

### A) Running Udava in Docker


```
docker build -t udava -f Dockerfile .
docker run -p 5000:5000 -it -v $(pwd)/assets:/usr/Udava/assets -v $(pwd)/.dvc:/usr/Udava/.dvc udava
```

### B) Run Udava directly on host

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

## Usage

### GUI

The GUI is available at `http://localhost:5000/`.
Follow the instructions in the GUI to create models and upload data for inference.

### API

The API is available at `http://localhost:5000/`.

#### POST /infer

##### JSON

The input should look like this:

```
{
  "param": {
    "modeluid": "3a7ee233-2380-4420-9e1d-246932bdede4"
  },
  "scalar": {
    "headers": ["time", "memory_used"],
    "data": [
      [1718725511, 1201184768],
      [1718725514, 1201840128]
    ]
  }
}
```

Explanation:

- **`param.modeluid`**: The unique identifier (UUID) of the model used.
- **`scalar.headers`**: An array of strings representing the data columns (in this case, `time` and `memory_used`).
- **`scalar.data`**: A 2D array with each inner array representing a data point:
  - The first value is a Unix timestamp.
  - The second value is the memory used in bytes.

The JSON can be sent to the API using `curl`:

```
curl -X POST -H "Content-Type: application/json" -d @data.json http://172.17.0.2:5000/infer
```

##### CSV

CSV data can be sent to the API using `curl`:

```
curl http://localhost:5000/infer -F file=@assets/data/raw/entrust-highend/simulated_low_end_traces_from_high_end_device.csv -F model_id=151d2394-7654-4958-9e82-174c7198368c
```

Make sure that the CSV contains the same columns as the model expects.
