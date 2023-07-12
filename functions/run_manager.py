import os


def setup_run(run):
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    experiment_run_path = os.path.join("mlruns", experiment_id, run_id)
    model_path = os.path.join("mlruns", experiment_id,
                              run_id, "artifacts/model/model.pkl")
    return run_id, experiment_id, experiment_run_path, model_path
