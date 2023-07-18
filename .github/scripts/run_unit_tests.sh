#!/bin/bash

echo "Running unit tests..."
python -m unittest -v tests.test_model_creator
#python -m unittest -v tests.test_model_runner
#python -m unittest -v tests.test_data_loader