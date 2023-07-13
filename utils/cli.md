# all
python C:\git\mlflow_config\utils\print_files.py --start_directory C:\git\mlflow_config\ --filter *configs\* *functions\*  *.github\* main.py --ignore *\functions\__pycache__*  --list_files --write
# all excl github pipelines
python C:\git\mlflow_config\utils\print_files.py --start_directory C:\git\mlflow_config\ --filter *configs\* *functions\*  main.py --ignore *\functions\__pycache__*  --list_files --write
# config only
python C:\git\mlflow_config\utils\print_files.py --start_directory C:\git\mlflow_config\ --filter *configs\* --list_files --write