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