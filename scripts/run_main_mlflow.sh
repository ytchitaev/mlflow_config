#!/bin/bash

python main_mlflow.py -e iris_tuning.json
python main_mlflow.py -e iris_no_tuning.json
python main_mlflow.py -e wine_tuning.json
python main_mlflow.py -e wine_no_tuning.json
