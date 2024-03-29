# 2023-08-02
 - https://chat.openai.com/share/fb0c46b8-b859-4bd7-a653-8818cb9a37e1
 - https://bytepawn.com/automatic-mlflow-logging-for-pytorch.html 
 - Not config driven
 - Use functions as generalisable functions, but don't use config file
 - Don't write execution data to config, just use the mlflow run ids
 - Could still use metrics evaluator for non autolog outputs
 - Could still use extensions but need to somehow confirm that related artefacts / python objects are created
 - Tuning runner is probably not necessary, can simply create examples to copy


# Notes3

- Stages could be a classmethod function e.g. Stage.initiate_run() instead of run_stage_initiate_run()
    - https://chat.openai.com/share/40fc1b06-78e5-4a2a-b6f6-4180035061aa
- Redo as initial scope:
    - Keep stages:
        - Stage.setup_experiment() - before with mlflow.start_run(), init autolog and run set_experiment()
        - Stage.initiate_run() - create python logger, setup tags, log configs
        - Stage.load_data()
        - Stage.split_data()
    - Move into a separate py:
        - 
# Notes2

- create_model in model_creator should be ModelCreator class, fit_model, log_model should be functions of the class
- Reduce hard coding of extension name between main_extensions and the plot_commands.
- Look at generalising load json / load csv back to the extension_runner main function
- Look at recording artifacts like this to allow individually storing name and extension separately

artifacts": {
        "cv_results" : {"extension" : "csv"},
        "best_estimator_evals_result" : {"extension" : "json"}
    }

# Notes
- can have various orchestrators that are pre-set for certain mdoels
- improve how extensions are processed, need a way to track artifacts - maybe write to config?
- artifacts in config are currently unused, either build this dynamically per above or remove
- callbacks are not currently implemented, either remove from config or adapt

Add support for:
{
"early_stopping": {
"stopping_rounds": 10,
"verbose": false
}

# Notes
- /function/*file should be grouped by an executing input - in future these can be orchestrated as required
# LATEST

- TODO - log artifacts should be made dynamic log_all_artifacts()
- It should be scoped to args - get_config(cfg, 'tuning.artifacts'), get_config(cfg, 'global') so that get_config can not be called from artifact_logger
- 'input_type' and 'output_type' should specify which mlflow_log_artifact_*_* method to use, maybe split read and write?
- The artifacts from tuning_result that are logged, should be dynamic based BOTH on (1) artifacts specified in config and (2) data class attribute not being none
- If (1) but not (2) specific log message should be generated
- TODO - Supress the stdout logging when I am redirecting to INFO
- TODO - Supress multi_logloss output if not verbose

# Main
- [x] Use this approach for building dynamic additional_args to pass to load_data() and similar approach for other such elements https://chat.openai.com/share/d563ff72-3ad0-454e-be70-70d984d6b52f
- [x] Move out processing of run_id / experiment_id / determining paths to a separate function / py file
- [x] passing of e.g. evaluation metrics that need to be logged should be done through classes that accumulate a register of metric/data subsets
- [x] evaluation_metrics dynamically from metadata based on metric + subset of data
- [x] mlflow.log_metric dynamically based on above
- [ ] move grid search to a separate py function file even if it is a single implementation, cv params should be optional
- [ ] bagging warning for regression - https://chat.openai.com/share/9ced9df8-9e17-40bb-85c1-942007711419 - lightgbm issue - https://github.com/microsoft/LightGBM/issues/5332
- [ ] look at how to abstract lightgbm and provide compatiability with basic pytorch implementation, may need to do it through python functions not config - https://chat.openai.com/share/ebb6bbca-63f5-40b7-b240-67c5b96299cc
- [ ] query for more improvements - data_load convert to factory - https://chat.openai.com/share/1286bdae-797d-4b08-b221-cc6ad90d5ae5
- [ ] I want to be able to pass callbacks dynamically from a mapping - callbacks=[lgb.log_evaluation(period=100, show_stdv=True)]
# Extensions
- [ ] Add `extensions` concept - tree diagram, chart metrics e.g. CV, launch predict ui
- [ ] Port over diagram drawing of gradient boosted tree
- [ ] Research how to plot data over time in mlflow, watch some videos
- [ ] Port over plotting evaluation metrics on sample or should that be visible on mlflow?
- [ ] Port over plotting cv_results or should that be visible in mflow?
- [ ] Best practice for sampling model predictions, maybe web ui or can mlflow do it?