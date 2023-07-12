
EXPERIMENT_MAPPING = {
    # setup
    'setup': {},
    'setup.experiment_name': 'Default',
    'setup.paths.run_temp_folder': 'temp',
    # data
    'data': {},
    'data.data_source': {},
    'data.dataset_name': {},
    'data.input_columns': [],
    'data.output_columns': [],
    # split
    'split': {},
    'split.train_percentage': 80,
    'split.validation_percentage': 10,
    'split.test_percentage': 10,
    'split.random_state': 42,
    # model
    'model': {},
    'model.library_name': {},
    'model.model_name': {},
    # tuning
    'tuning': {},
    'tuning.grid_search': {},
    'tuning.grid_search.param_grid': {},
    'tuning.grid_search.cv': {},
    # evaluate
    'evaluate': {},
    # artefacts
    'artefacts': {},
    'artefacts.python_logging.file_name': 'python_logging.txt',
    'artefacts.cv_results.file_name': 'cv_results.csv',

    ## This is an optional list. If it's not present in the config, it won't cause an error
    #'setup.optional_list': [],

}

GLOBAL_MAPPING = {
    "py_log_file_name": "python_logging.txt",
    "run_temp_subdir": "temp"
}