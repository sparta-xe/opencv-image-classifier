#!/usr/bin/env bash
set -e

# Create venv if absent
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Could not find .venv/bin/activate. Create the venv manually."
  exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt

# Train with default classes if data exists
if [ -d "data/cats" ] && [ -d "data/dogs" ]; then
  python src/train.py --data-root data --classes cats dogs --img-size 64 --model-out models/image_classifier.joblib
  echo "Try prediction with: python src/predict.py --image path/to/image.jpg"
else
  echo "Place your images under data/<class_name>/ before training."
fi