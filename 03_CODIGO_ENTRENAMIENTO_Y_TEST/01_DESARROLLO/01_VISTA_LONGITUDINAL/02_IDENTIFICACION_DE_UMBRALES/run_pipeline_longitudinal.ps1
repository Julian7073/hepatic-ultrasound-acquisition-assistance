param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [string]$ScriptPath
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "Script: $ScriptPath" -ForegroundColor DarkCyan
    Write-Host "============================================================" -ForegroundColor Cyan

    & $Python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo el paso: $Title"
    }
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "Proyecto: $ProjectRoot" -ForegroundColor Green
Write-Host "Python: $Python" -ForegroundColor Green

Run-Step "01 - Auditar dataset Roboflow COCO" ".\Codigos_Segmentacion_Longitudinal\01_auditar_dataset_roboflow.py"
Run-Step "02 - Generar mascaras desde COCO" ".\Codigos_Segmentacion_Longitudinal\02_generar_mascaras_coco.py"
Run-Step "03 - Generar overlays de control visual" ".\Codigos_Segmentacion_Longitudinal\03_qc_visual_mascaras.py"
Run-Step "04 - Calcular metricas GLCM longitudinales" ".\Codigos_Segmentacion_Longitudinal\04_pipeline_glcm_longitudinal.py"
Run-Step "05 - Definir umbrales iniciales" ".\Codigos_Segmentacion_Longitudinal\05_definir_umbral_aceptabilidad.py"
Run-Step "08 - Analizar aceptabilidad longitudinal" ".\Codigos_Segmentacion_Longitudinal\08_analizar_resultados_longitudinales.py"
Run-Step "10 - Generar anexo visual longitudinal" ".\Codigos_Segmentacion_Longitudinal\10_generar_anexo_visual_longitudinal.py"
Run-Step "11 - Ajustar regla con borde y contraste local" ".\Codigos_Segmentacion_Longitudinal\11_ajustar_regla_aceptabilidad_borde.py"

Write-Host ""
Write-Host "Pipeline longitudinal completado correctamente." -ForegroundColor Green
Write-Host "Salidas principales:" -ForegroundColor Green
Write-Host "- outputs\metrics\glcm_longitudinal_metrics_with_border_rule.csv"
Write-Host "- outputs\reports\longitudinal_acceptability_border_rule_summary.csv"
Write-Host "- outputs\reports\longitudinal_doubtful_cases_review_list.csv"
Write-Host "- outputs\figures\doubtful_cases_top20.png"
