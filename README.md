# OpenCV Image Classifier (Cats vs Dogs example)

A minimal, **ready-to-run** image classification project that uses **OpenCV** for image loading/resizing and **scikit-learn** (SVM) for classification. Works for any 2–3 (or more) classes as long as you place your images into per-class folders.

---

## 📦 Project Structure

```
opencv-image-classifier/
│── data/                 # Put your class folders here (e.g., cats/, dogs/)
│── models/               # Saved models and label map
│── notebooks/            # Optional: Jupyter notebooks
│── scripts/              # Helper scripts
│── src/                  # Source code
│   ├── preprocess.py     # image loading & dataset utilities
│   ├── train.py          # training script (SVM)
│   └── predict.py        # prediction script (CLI)
│── requirements.txt      # dependencies
│── .gitignore            # ignores venv, model files, data, etc.
│── README.md             # this file
```

> Expected data layout (example):
>
> ```
> data/
> ├── cats/
> │   ├── cat001.jpg
> │   ├── cat002.jpg
> │   └── ...
> └── dogs/
>     ├── dog001.jpg
>     ├── dog002.jpg
>     └── ...
> ```

---

## 🚀 Quickstart

### 0) Clone or download this project
If you downloaded a zip, extract it and `cd` into the folder.

```bash
cd opencv-image-classifier
```

### 1) (Recommended) Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Prepare your dataset
Place images into class folders under `data/`. For example:
```
data/cats/*.jpg
data/dogs/*.jpg
```

### 4) Train
Default classes are `cats` and `dogs`. Customize as needed.

**macOS / Linux / Windows (same command):**
```bash
python src/train.py --data-root data --classes cats dogs --img-size 64 --model-out models/image_classifier.joblib
```

What this does:
- Loads images, converts to grayscale, resizes to 64×64
- Normalizes pixel values to [0,1]
- Splits into train/test (80/20)
- Trains a linear SVM classifier
- Saves the model to `models/image_classifier.joblib`
- Saves the label map to `models/labels.json`

### 5) Predict on a single image

```bash
python src/predict.py --image path/to/your/test_image.jpg --model models/image_classifier.joblib --labels models/labels.json --img-size 64
```

### 6) (Optional) One-command setup & train
A convenience script that installs dependencies and runs training if you already placed data:
```bash
bash scripts/quickstart.sh
```
Windows PowerShell:
```powershell
./scripts/quickstart.ps1
```

---

## 📋 Command Reference

**Training**
```
python src/train.py   --data-root data   --classes cats dogs   --img-size 64   --test-size 0.2   --kernel linear   --model-out models/image_classifier.joblib   --random-state 42
```

**Prediction**
```
python src/predict.py   --image path/to/image.jpg   --model models/image_classifier.joblib   --labels models/labels.json   --img-size 64
```

> You can add more classes: `--classes apples bananas oranges` provided you have matching folders in `data/`.

---

## 🧠 Notes & Tips

- If images vary in orientation/quality, consider adding **augmentation** or using **HOG** features or a CNN for better performance.
- If your dataset is small, try `--kernel rbf` and tune `--C` and `--gamma` on the SVM.
- Ensure roughly balanced number of images per class.
- Large images are automatically resized; you can change `--img-size`.

---

## 🧪 Reproducibility

- Scripts use a fixed `--random-state` by default.
- Exact results may still vary if your data changes.

---

## 🔧 Troubleshooting

- **ImportError: No module named cv2** → `pip install opencv-python`
- **Not enough images** → Ensure you have at least a few dozen images per class.
- **Poor accuracy** → Try higher `--img-size` (e.g., 128), `--kernel rbf`, or clean up mislabeled images.

---

## 🐙 GitHub Setup (optional)

1. Go to https://github.com
2. New Repository → name it (e.g., `opencv-image-classifier`)
3. Select **Public**, add README, add **.gitignore: Python**, License: **MIT**
4. Locally in this folder:
   ```bash
   git init
   git remote add origin https://github.com/USERNAME/opencv-image-classifier.git
   git add .
   git commit -m "Initial commit - OpenCV Image Classifier"
   git branch -M main
   git push -u origin main
   ```

---

## 📚 License
MIT