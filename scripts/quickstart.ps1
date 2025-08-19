param()

$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv")) {
    python -m venv .venv
}

$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    & $activate
} else {
    Write-Error "Could not find .venv\Scripts\Activate.ps1. Create the venv manually."
}

python -m pip install --upgrade pip
pip install -r requirements.txt

if ((Test-Path ".\data\cats") -and (Test-Path ".\data\dogs")) {
    python src\train.py --data-root data --classes cats dogs --img-size 64 --model-out models\image_classifier.joblib
    Write-Host "Try prediction with: python src\predict.py --image path\to\image.jpg"
} else {
    Write-Host "Place your images under data\<class_name>\ before training."
}