```md
python print_files.py --start_directory C:/git/gradient_boosting/ --extension .py --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --extension .py --list_files --recursive
python print_files.py --start_directory C:/git/gradient_boosting/ --extension .py --list_files --recursive --ignore *env\*
python print_files.py --start_directory C:/git/gradient_boosting/ --extension .py --write --recursive --ignore *env\*
python print_files.py --start_directory C:/git/gradient_boosting/ --filter lgbm_regression.py main.py spec.json --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive\v3\main.py* *archive\v3\predict.py* --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive\v1\* *archive\v2\* --extension .py --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive\v1\* --extension .py --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive\v1\* *archive\v2\*  --extension .py --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive/v3/main.py* *archive/v3/predict.py* --write --list_files
python print_files.py --start_directory C:/git/gradient_boosting/ --filter *archive/v3/main.py* *archive/v3/predict.py* --write
python C:\git\mlflow\utils\print_files.py --start_directory C:\git\mlflow\ --filter config.json load_config.py load_dataset.py main.py --list_files 
python C:\git\mlflow\utils\print_files.py --start_directory C:\git\mlflow\ --filter config.json load_config.py load_dataset.py main.py --write
python C:\git\mlflow\utils\print_files.py --start_directory C:\git\mlflow\ --filter config.json load_config.py load_dataset.py run_model.py run_logging.py main.py --list_files 
python C:\git\mlflow\utils\print_files.py --start_directory C:\git\mlflow\ --filter config.json load_config.py load_dataset.py run_model.py run_logging.py main.py --write

python C:\git\mlflow_config\utils\print_files.py --start_directory C:\git\mlflow_config\ --filter config.json config_loader.py data_loader.py evaluation_metrics.py main.py mlflow_logger.py model_runner.py --write --list_files 

python C:\git\mlflow_config\utils\print_files.py --start_directory C:\git\mlflow_config\ --filter *functions\config.json* *functions\config_loader.py* *functions\data_loader.py* *functions\evaluation_metrics.py* main.py *functions\mlflow_logger.py* *functions\model_runner.py* --write --list_files 
```