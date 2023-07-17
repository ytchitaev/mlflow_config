# all
python C:\git\mlflow_config\cgpt\print_files.py `
--start_directory C:\git\mlflow_config\ `
--filter *configs\* *functions\* *utils\* *.github\* *extensions\* *outputs\* *main_mlflow.py *main_extensions.py `
--ignore *\functions\__pycache__* *\utils\__pycache__* *\extensions\__pycache__* *\env* `
--list_files `
--write
# exclude github actions
python C:\git\mlflow_config\cgpt\print_files.py `
--start_directory C:\git\mlflow_config\ `
--filter *configs\* *functions\* *utils\* *extensions\* *outputs\* *main_mlflow.py *main_extensions.py `
--ignore *\functions\__pycache__* *\utils\__pycache__* *\extensions\__pycache__* *\env* `
--list_files `
--write
# extensions only
python C:\git\mlflow_config\cgpt\print_files.py `
--start_directory C:\git\mlflow_config\ `
--filter *utils\* *extensions\* *outputs\* *main_extensions.py `
--ignore *\functions\__pycache__* *\utils\__pycache__* *\extensions\__pycache__* *\env* `
--list_files `
--write



# all excl github pipelines
python C:\git\mlflow_config\cgpt\print_files.py --start_directory C:\git\mlflow_config\ --filter *configs\* *functions\*  main_mlflow.py --ignore *\functions\__pycache__*  --list_files --write
# config only
python C:\git\mlflow_config\cgpt\print_files.py --start_directory C:\git\mlflow_config\ --filter *configs\* --list_files --write