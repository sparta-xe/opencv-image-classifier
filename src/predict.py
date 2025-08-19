import os
import json
import argparse
import cv2
import numpy as np
from joblib import load

def preprocess_image(image_path: str, img_size=(64,64), grayscale=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    img = cv2.resize(img, img_size)
    if grayscale:
        arr = img.astype('float32') / 255.0
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = (img.astype('float32') / 255.0)
    return arr.flatten().reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Predict a single image with a trained SVM model.")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/image_classifier.joblib', help='Path to trained model')
    parser.add_argument('--labels', type=str, default='models/labels.json', help='Path to labels JSON')
    parser.add_argument('--img-size', type=int, default=64, help='Image resize (square) used during training')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.isfile(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    model = load(args.model)
    with open(args.labels, 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    X = preprocess_image(args.image, img_size=(args.img_size, args.img_size), grayscale=True)
    pred = model.predict(X)[0]
    label = label_map.get(int(pred), str(pred))
    print(label)

if __name__ == '__main__':
    main()