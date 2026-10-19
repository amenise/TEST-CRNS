# Runs every GPU experiment sequentially (one GPU; parallel jobs overflow the 4 GB card). Resumable: re-running skips
# finished runs. Logs in runs/logs/. Usage: powershell -File scripts/run_pipeline.ps1
# Reduced budget (~4 h on an RTX 2050): see configs/tta.yaml and configs/ctta.yaml.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force runs\logs | Out-Null
$py = ".venv\Scripts\python.exe"
$steps = @(
    @("hp_vit", "-m birdshift.cli.select_hp --backbone vit_s16"),
    @("tta_vit", "-m birdshift.cli.run_tta --backbone vit_s16"),
    @("analyze_tta_vit", "-m birdshift.cli.analyze_tta --backbone vit_s16"),
    @("ctta_vit", "-m birdshift.cli.run_ctta --backbone vit_s16 --sequences brief year"),
    @("ctta_long_vit", "-m birdshift.cli.run_ctta --backbone vit_s16 --sequences brief_long"),
    @("analyze_ctta_vit", "-m birdshift.cli.analyze_ctta --backbone vit_s16"),
    @("hp_resnet", "-m birdshift.cli.select_hp --backbone resnet50"),
    @("tta_resnet", "-m birdshift.cli.run_tta --backbone resnet50"),
    @("analyze_tta_resnet", "-m birdshift.cli.analyze_tta --backbone resnet50"),
    @("failure_resnet", "-m birdshift.cli.run_failure --backbone resnet50 --parts batch noniid"),
    @("failure_vit", "-m birdshift.cli.run_failure --backbone vit_s16 --parts noniid")
)
foreach ($s in $steps) {
    $t0 = Get-Date
    "[$t0] START $($s[0])" | Tee-Object -Append runs\logs\pipeline.log
    $p = Start-Process -FilePath $py -ArgumentList $s[1] -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "runs\logs\$($s[0]).out" -RedirectStandardError "runs\logs\$($s[0]).err"
    "[$(Get-Date)] END $($s[0]) exit=$($p.ExitCode) ($([int]((Get-Date) - $t0).TotalMinutes) min)" | Tee-Object -Append runs\logs\pipeline.log
}
