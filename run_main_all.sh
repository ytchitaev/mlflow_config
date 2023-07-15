#!/bin/bash

#categorisation
python main_mlflow.py -e iris_tuning.json
python main_extensions.py
python main_mlflow.py -e iris_no_tuning.json
python main_extensions.py

#regression
python main_mlflow.py -e wine_tuning.json
python main_extensions.py
python main_mlflow.py -e wine_no_tuning.json
python main_extensions.py
