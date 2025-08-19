import os
import cv2
import numpy as np
from typing import List, Tuple, Dict

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS

def load_images_from_folder(folder: str, label: int, img_size: Tuple[int, int]=(64, 64), grayscale: bool=True):
    """Load images from a single folder, return list of (image_array, label)."""
    data = []
    for filename in os.listdir(folder):
        if not is_image_file(filename):
            continue
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            # Skip unreadable files
            continue
        img = cv2.resize(img, img_size)
        if grayscale:
            # img shape: (H, W)
            arr = img.astype('float32') / 255.0
        else:
            # BGR -> RGB then normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = (img.astype('float32') / 255.0).reshape(-1)
        data.append((arr, label))
    return data

def load_dataset(data_root: str, class_names: List[str], img_size: Tuple[int, int]=(64, 64), grayscale: bool=True):
    """Load dataset given a root and list of class folder names.

    Returns:
        X: np.ndarray [N, F]
        y: np.ndarray [N]
        paths: List[str]
        label_map: Dict[int, str]
    """
    X_list = []
    y_list = []
    paths = []
    label_map = {}
    for idx, cls in enumerate(class_names):
        label_map[idx] = cls
        folder = os.path.join(data_root, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for filename in os.listdir(folder):
            if not is_image_file(filename):
                continue
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            if grayscale:
                arr = img.astype('float32') / 255.0
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                arr = (img.astype('float32') / 255.0)
            X_list.append(arr.flatten())
            y_list.append(idx)
            paths.append(img_path)
    if not X_list:
        raise RuntimeError("No images were loaded. Check your data folders and file extensions.")
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y, paths, label_map