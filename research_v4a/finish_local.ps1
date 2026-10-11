param(
    [Parameter(Mandatory=$true)][string]$Python,
    [int]$BenchmarkPid = 0,
    [int]$InitialConfirmationPid = 0
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function Wait-Existing([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
}
function Invoke-Stage([string[]]$Arguments) {
    & $Python -B -m @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment stage failed: $Arguments" }
}
# The controller does no statistical computation while waiting. At most one
# initial confirmation process and one child Python process coexist.
Wait-Existing $BenchmarkPid
Invoke-Stage @('research_v4a.repair_decisions','development')
Invoke-Stage @('research_v4a.repair_decisions','confirmation')
Wait-Existing $InitialConfirmationPid
$manifest = Get-Content -LiteralPath 'research_v4a/results/S2_confirmation_validated_manifest.json' -Raw | ConvertFrom-Json
if (-not $manifest.complete) { Invoke-Stage @('research_v4a.repair_decisions','confirmation') }
Invoke-Stage @('research_v4a.posterior_precision')
Invoke-Stage @('research_v4a.refine_certificates')
# Retain the original error-pressure experiment, then repeat the fixed cases
# with the same numerical repair as the benchmark and confirmation decisions.
$pressure = Join-Path $PSScriptRoot 'results/S3_perturbation.jsonl'
$pressureInitial = Join-Path $PSScriptRoot 'results/S3_perturbation_initial.jsonl'
if (-not (Test-Path -LiteralPath $pressureInitial)) {
    Move-Item -LiteralPath $pressure -Destination $pressureInitial
}
Invoke-Stage @('research_v4a.computation','perturbation','--numerical-repair')
Invoke-Stage @('research_v4a.secondary_tie_audit')
Invoke-Stage @('research_v4a.audit')
Invoke-Stage @('research_v4a.summarize')
Invoke-Stage @('research_v4a.delivery_check')
Write-Output 'Computation and analysis complete; inspect report and figures before archiving and committing.'
