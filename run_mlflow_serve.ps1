$jsonFile = Get-Content -Raw -Path "outputs/last_exec.json"
$lastRun = ConvertFrom-Json $jsonFile
$runId = $lastRun.run_id
$experimentId = $lastRun.experiment_id

$mlflowCommand = "mlflow models serve --model-uri runs:/$runId/model -h localhost -p 5001"
Invoke-Expression -Command $mlflowCommand
