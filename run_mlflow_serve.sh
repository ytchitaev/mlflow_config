#!/bin/bash

JSON_FILE="outputs/last_exec.json"
LAST_EXEC=$(cat $JSON_FILE)
RUN_ID=$(echo $LAST_EXEC | jq -r '.run_id')
EXPERIMENT_ID=$(echo $LAST_EXEC | jq -r '.experiment_id')

mlflow models serve --model-uri "runs:/$RUN_ID/model" -h localhost -p 5001
