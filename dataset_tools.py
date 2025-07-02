import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import cv2
from collections import Counter
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, Rotate,
    ShiftScaleRotate, RandomCrop, Resize, Compose
)
from tqdm import tqdm

def create_stratified_splits(data_dir, TRAIN_SPLIT, val_ratio, test_ratio, seed=42):
    """
    Create stratified train/val/test splits ensuring balanced class distribution
    """
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(class_path, img_file))
                    labels.append(class_to_idx[class_name])
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # First split: separate train from (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(val_ratio + test_ratio), 
        stratify=labels, 
        random_state=seed
    )
    
    # Second split: separate val from test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=1-val_test_ratio, 
        stratify=y_temp, 
        random_state=seed
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


def create_tf_dataset_from_paths(image_size, image_paths, labels, batch_size, seed, shuffle=True):
    """
    Create TensorFlow dataset from image paths and labels
    """
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    return dataset


def create_augmented_dataset(image_paths, labels, target_samples_per_class, output_base_dir, resize_to):
    os.makedirs(output_base_dir, exist_ok=True)

    # Define augmentation pipeline (efficient and compiled)
    augment_fn = Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=10, p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
        RandomBrightnessContrast(p=0.5),
        Resize(resize_to[0], resize_to[1])
    ])

    # Group images by class
    class_to_paths = {}
    for path, label in zip(image_paths, labels):
        class_to_paths.setdefault(label, []).append(path)

    augmented_paths = []
    augmented_labels = []

    for class_label, paths in class_to_paths.items():
        cur_count = len(paths)
        needed = max(0, target_samples_per_class - cur_count)
        print(f"\nClass {class_label}: Need {needed} more samples")

        class_output_dir = os.path.join(output_base_dir, f"class_{class_label}")
        os.makedirs(class_output_dir, exist_ok=True)

        # Add original images
        augmented_paths.extend(paths)
        augmented_labels.extend([class_label] * cur_count)

        if needed == 0:
            continue

        augs_per_image = (needed // cur_count) + 1
        base_names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        generated = 0

        for idx, (img_path, base_name) in enumerate(zip(paths, base_names)):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for j in range(augs_per_image):
                if generated >= needed:
                    break

                aug = augment_fn(image=image)['image']
                aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

                aug_path = os.path.join(class_output_dir, f"{base_name}_aug_{j}.jpg")
                cv2.imwrite(aug_path, aug_bgr)

                augmented_paths.append(aug_path)
                augmented_labels.append(class_label)
                generated += 1

            if generated >= needed:
                break

    print(f"\n✅ Dataset created with {len(augmented_paths)} total samples.")
    return np.array(augmented_paths), np.array(augmented_labels)