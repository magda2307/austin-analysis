param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dashboard", "pipeline", "quick", "test", "jupyter")]
    [string]$Command
)

switch ($Command) {
    "dashboard" { docker compose up dashboard }
    "pipeline"  { docker compose run --rm pipeline-full }
    "quick"     { docker compose run --rm pipeline-quick }
    "test"      { docker compose run --rm test }
    "jupyter"   { docker compose up jupyter }
}
