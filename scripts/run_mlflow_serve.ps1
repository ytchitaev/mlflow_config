$jsonFile = Get-Content -Raw -Path "outputs/last_execution_config.json"
$lastRun = ConvertFrom-Json $jsonFile
$runId = $lastRun.execution.run_id
$experimentId = $lastRun.execution.experiment_id

$mlflowCommand = "mlflow models serve --model-uri runs:/$runId/model -h localhost -p 5001"
Invoke-Expression -Command $mlflowCommand
