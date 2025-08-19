import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from preprocess import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Train an SVM image classifier with OpenCV + scikit-learn.")
    parser.add_argument('--data-root', type=str, default='data', help='Root folder containing class subfolders')
    parser.add_argument('--classes', type=str, nargs='+', default=['cats','dogs'], help='List of class folder names')
    parser.add_argument('--img-size', type=int, default=64, help='Square image size (pixels) for resizing')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear','rbf','poly','sigmoid'], help='SVM kernel')
    parser.add_argument('--C', type=float, default=1.0, help='SVM regularization parameter')
    parser.add_argument('--gamma', type=str, default='scale', help='SVM gamma (for rbf/poly/sigmoid)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--model-out', type=str, default='models/image_classifier.joblib', help='Path to save trained model')
    parser.add_argument('--labels-out', type=str, default='models/labels.json', help='Path to save label mapping')
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    print(f"Loading dataset from '{args.data_root}' with classes: {args.classes}")
    X, y, paths, label_map = load_dataset(args.data_root, args.classes, img_size=img_size, grayscale=True)
    print(f"Loaded {len(y)} images. Feature dim: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"Training SVM (kernel={args.kernel}, C={args.C}, gamma={args.gamma})...")
    model = SVC(kernel=args.kernel, C=args.C, gamma=args.gamma)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())]))

    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_out), exist_ok=True)

    print(f"Saving model to {args.model_out}")
    dump(model, args.model_out)

    print(f"Saving label map to {args.labels_out}")
    with open(args.labels_out, 'w') as f:
        json.dump(label_map, f, indent=2)

    print("Done.")

if __name__ == '__main__':
    main()