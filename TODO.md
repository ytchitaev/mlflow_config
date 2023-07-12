# Main
- [x] Use this approach for building dynamic additional_args to pass to load_data() and similar approach for other such elements https://chat.openai.com/share/d563ff72-3ad0-454e-be70-70d984d6b52f
- [x] Move out processing of run_id / experiment_id / determining paths to a separate function / py file
- [x] passing of e.g. evaluation metrics that need to be logged should be done through classes that accumulate a register of metric/data subsets
- [x] evaluation_metrics dynamically from metadata based on metric + subset of data
- [x] mlflow.log_metric dynamically based on above
- [ ] move grid search to a separate py function file even if it is a single implementation
- [ ] bagging warning for regression - https://chat.openai.com/share/9ced9df8-9e17-40bb-85c1-942007711419
- [ ] look at how to abstract lightgbm and provide compatiability with basic pytorch implementation, may need to do it through python functions not config - https://chat.openai.com/share/ebb6bbca-63f5-40b7-b240-67c5b96299cc
# Other
- [ ] Port over diagram drawing of gradient boosted tree
- [ ] Research how to plot data over time in mlflow, watch some videos
- [ ] Port over plotting evaluation metrics on sample or should that be visible on mlflow?
- [ ] Port over plotting cv_results or should that be visible in mflow?
- [ ] Best practice for sampling model predictions, maybe web ui or can mlflow do it?
