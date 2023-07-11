# Main
- [ ] Use this approach for building dynamic additional_args to pass to load_data() and similar approach for other such elements https://chat.openai.com/share/d563ff72-3ad0-454e-be70-70d984d6b52f
- [ ] Move out processing of run_id / experiment_id / determining paths to a separate function / py file
- [ ] passing of e.g. evaluation metrics that need to be logged should be done through classes that accumulate a register of metric/data subsets
- [ ] evaluation_metrics dynamically from metadata based on metric + subset of data
- [ ] mlflow.log_metric dynamically based on above
- [ ] bagging warning for regression - https://chat.openai.com/share/9ced9df8-9e17-40bb-85c1-942007711419

# Other
- [ ] Port over diagram drawing of gradient boosted tree
- [ ] Research how to plot data over time in mlflow, watch some videos
- [ ] Port over plotting evaluation metrics on sample or should that be visible on mlflow?
- [ ] Port over plotting cv_results or should that be visible in mflow?
- [ ] Best practice for sampling model predictions, maybe web ui or can mlflow do it?

