# mcv-c6-2025-team5 Week 4
## Project slides
The google slides for this MTMC tracking can be found [here](https://docs.google.com/presentation/d/1T8Tt-Z8W9OkqIi9D2IEpYVQgj3iiGqniKY6EPJmwFf8/edit#slide=id.g33e8f63fe75_0_826)

## Object detection:



## MOT:

## MTMC:

## Metrics:
For computing the metric the MTMC configuration already has the HOTA and IDF1, amongs other metrics, in the global apporximation instead of average. For the metric in average of the camera sequences we can execute the following script:
```
cd  vehicle_mtmc/
export PYTHONPATH=$(pwd)
python evaluate/hota.py --pred "output/cityflow_s03" parser.add_argument("--gt", default="/home/marco/Downloads/aic19-track1-mtmc-train/train/S03/"
```
