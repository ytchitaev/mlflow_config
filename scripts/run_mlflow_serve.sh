#!/bin/bash

JSON_FILE="outputs/last_execution_config.json"
LAST_EXEC=$(cat $JSON_FILE)
RUN_ID=$(echo $LAST_EXEC | jq -r 'execution.run_id')
EXPERIMENT_ID=$(echo $LAST_EXEC | jq -r 'execution.experiment_id')

mlflow models serve --model-uri "runs:/$RUN_ID/model" -h localhost -p 5001
