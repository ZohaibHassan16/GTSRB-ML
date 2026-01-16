import os
import logging
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import shutil
import random
import math
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import time

# importing gradio for front-end stuff
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Setting up Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', force=True)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Central configuration for reproducibility."""
    PROJECT_NAME = "GTSRB_TRAFFIC_SIGN"

    # Paths 
    BASE_DIR = Path(".")
    DATA_ROOT = BASE_DIR / "data"
    TRAIN_DIR = DATA_ROOT / "Train"
    TEST_DIR = DATA_ROOT / "Test"
    MODELS_DIR = BASE_DIR / "models"
    
    # Zip locs if downloading data from the script provided
    
    ZIP_PATH = DATA_ROOT / "Train.zip"
    TEST_ZIP_PATH = DATA_ROOT / "Test.zip"
    TEST_CSV_PATH = DATA_ROOT / "Test.csv"
    
    SIGN_NAMES_URL = "https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/master/signnames.csv"

    IMG_SIZE = (160, 160)
    NUM_CHANNELS = 3
    NUM_CLASSES = 43
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    SEED = 42

    # Model Regularization
    DROPOUT_RATE = 0.2
    FINE_TUNE_AT = 50

    def __post_init__(self):
        self.INPUT_SHAPE = (*self.IMG_SIZE, self.NUM_CHANNELS)
        self.DATA_ROOT.mkdir(exist_ok=True, parents=True)
        self.MODELS_DIR.mkdir(exist_ok=True, parents=True)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)

class GTSRBDataManager:
    """Handles data downloading, extraction and dataset creation."""

    def __init__(self, config):
        self.cfg = config
        self.class_names = None

    def setup_data(self):
        """Organizing data if necessary."""
        # Training Data
        if not self.cfg.TRAIN_DIR.exists():
            if self.cfg.ZIP_PATH.exists():
                logger.info(f"Unzipping Train data from {self.cfg.ZIP_PATH}...")
                with zipfile.ZipFile(self.cfg.ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(self.cfg.DATA_ROOT)
                logger.info("Unzipping Train complete.")
            else:
                logger.warning(f"Train Zip not found at {self.cfg.ZIP_PATH}. Ensure data is placed in {self.cfg.DATA_ROOT}")

        # Test Data
        test_has_images = self.cfg.TEST_DIR.exists() and len(list(self.cfg.TEST_DIR.glob("*.png"))) > 0

        if not test_has_images:
            if self.cfg.TEST_ZIP_PATH.exists():
                logger.info(f"Unzipping Test data from {self.cfg.TEST_ZIP_PATH}...")
                with zipfile.ZipFile(self.cfg.TEST_ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(self.cfg.DATA_ROOT)

                loose_images = list(self.cfg.DATA_ROOT.glob("*.png"))
                if loose_images:
                    logger.info(f"Found {len(loose_images)} loose images. Moving to {self.cfg.TEST_DIR}...")
                    self.cfg.TEST_DIR.mkdir(exist_ok=True)
                    for img in loose_images:
                        shutil.move(str(img), str(self.cfg.TEST_DIR / img.name))
                    logger.info("Test data organized successfully.")
            else:
                logger.warning(f"Test Zip not found at {self.cfg.TEST_ZIP_PATH}")

        # Downloading labels
        label_file = self.cfg.DATA_ROOT / "signnames.csv"
        if not label_file.exists():
            try:
                logger.info("Downloading signnames.csv...")
                response = requests.get(self.cfg.SIGN_NAMES_URL)
                response.raise_for_status()
                with open(label_file, 'w') as f:
                    f.write(response.text)
            except Exception as e:
                logger.error(f"Failed to download labels : {e}")

        # Copying Test CSV if it was extracted to parent of zip
        if self.cfg.TEST_ZIP_PATH.exists():
            drive_csv = self.cfg.TEST_ZIP_PATH.parent / "Test.csv"
            local_csv = self.cfg.DATA_ROOT / "Test.csv"
            if drive_csv.exists() and not local_csv.exists():
                 shutil.copy(drive_csv, local_csv)

    def load_signnames_csv(self):
        label_path = self.cfg.DATA_ROOT / "signnames.csv"
        if not label_path.exists(): return {}
        try:
            df = pd.read_csv(label_path)
            return dict(zip(df["ClassId"], df["SignName"]))
        except Exception: return {}

    def get_datasets(self, preprocess_func=None):
        """Creates optimized training and validation datasets."""
        if not self.cfg.TRAIN_DIR.exists():
            # Attempt setup if directory missing
            self.setup_data()
            if not self.cfg.TRAIN_DIR.exists():
                raise FileNotFoundError(f"Training directory not found: {self.cfg.TRAIN_DIR}")

        logger.info("Creating datasets from directory...")

        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            if preprocess_func:
                image = preprocess_func(image)
            else:
                image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image, label

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.cfg.TRAIN_DIR, validation_split=0.2, subset="training",
            seed=self.cfg.SEED, image_size=self.cfg.IMG_SIZE,
            batch_size=self.cfg.BATCH_SIZE, label_mode='categorical'
        )

        self.class_names = train_ds.class_names

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.cfg.TRAIN_DIR, validation_split=0.2, subset="validation",
            seed=self.cfg.SEED, image_size=self.cfg.IMG_SIZE,
            batch_size=self.cfg.BATCH_SIZE, label_mode='categorical'
        )

        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.map(preprocess, num_parallel_calls=autotune).shuffle(200).prefetch(autotune)
        val_ds = val_ds.map(preprocess, num_parallel_calls=autotune).prefetch(autotune)
        return train_ds, val_ds

    def get_test_dataset(self, preprocess_func=None):
        """Loads the test dataset"""
        if not self.cfg.TEST_CSV_PATH.exists(): 
            logger.warning(f"Test CSV not found at {self.cfg.TEST_CSV_PATH}")
            return None
        
        # We need class names to map labels correctly
        if self.class_names is None:
            # inf. from train dir if not set
            if self.cfg.TRAIN_DIR.exists():
                self.class_names = sorted([d.name for d in self.cfg.TRAIN_DIR.iterdir() if d.is_dir()])
            else:
                logger.error("Cannot load test dataset without class names (run get_datasets first or ensure Train dir exists).")
                return None

        logger.info(f"Loading test data from {self.cfg.TEST_CSV_PATH}...")
        df = pd.read_csv(self.cfg.TEST_CSV_PATH)

        folder_to_index = {name: i for i, name in enumerate(self.class_names)}
        def remap_label(real_id):
            return folder_to_index.get(str(real_id), 0)

        valid_paths = []
        valid_labels = []

        # Optimization
        for idx, row in df.iterrows():
            p1 = self.cfg.DATA_ROOT / row['Path']
            p2 = self.cfg.DATA_ROOT / "Test" / os.path.basename(row['Path'])
            
            final_path = None
            if p1.exists(): final_path = str(p1)
            elif p2.exists(): final_path = str(p2)

            if final_path:
                valid_paths.append(final_path)
                valid_labels.append(remap_label(row['ClassId']))

        if not valid_paths: 
            logger.warning("No valid test images found based on CSV paths.")
            return None

        logger.info(f"Found {len(valid_paths)} valid test images.")
        ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.io.decode_png(img, channels=3)
            img = tf.image.resize(img, self.cfg.IMG_SIZE)
            img = tf.cast(img, tf.float32)
            if preprocess_func:
              img = preprocess_func(img)
            else:
              img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            return img, tf.one_hot(label, self.cfg.NUM_CLASSES)

        autotune = tf.data.AUTOTUNE
        ds = ds.map(load_and_preprocess, num_parallel_calls=autotune)
        ds = ds.batch(self.cfg.BATCH_SIZE).prefetch(autotune)
        return ds

class EDAExplorer:
    """Handles EDA tasks."""

    def __init__(self, config: 'Config', label_map: 'Dict[int, str]'):
        self.cfg = config
        self.label_map = label_map

    def plot_class_distribution(self):
        """Plots distribution of classes."""
        logger.info("Analyzing class distribution...")

        if not self.cfg.TRAIN_DIR.exists():
          logger.error(f"Train directory not found")
          return
        class_counts = {}

        for class_id in os.listdir(self.cfg.TRAIN_DIR):
            class_dir = self.cfg.TRAIN_DIR / class_id
            if class_dir.is_dir():
                try:
                    cid = int(class_id)
                    images = list(class_dir.glob('*.*'))
                    valid_images = [f for f in images if f.suffix.lower() in [".png", ".jpg", ".ppm"]]
                    count = len(valid_images)
                    class_counts[cid] = count
                except ValueError:
                  continue

        if not class_counts:
            logger.warning("No images found.")
            return

        df = pd.DataFrame(list(class_counts.items()), columns=["ClassId", "Count"])
        df["Name"] = df["ClassId"].map(self.label_map)
        df = df.sort_values('Count', ascending=False)

        plt.figure(figsize=(12, 12))
        sns.barplot(data=df, y="Name", x="Count", hue="Name", palette='viridis', legend=False)
        plt.title("Distribution of Traffic signs in Training Set")
        plt.tight_layout()
        plt.show()

    def visualize_samples(self, num_samples=15):
        """Visualizes a grid of random samples from the training set."""
        logger.info("Visualizing random samples...")
        if not self.cfg.TRAIN_DIR.exists(): return
        
        all_classes = [d for d in os.listdir(self.cfg.TRAIN_DIR) if (self.cfg.TRAIN_DIR / d).is_dir()]
        selected_classes = np.random.choice(all_classes, min(len(all_classes), num_samples), replace=False)

        plt.figure(figsize=(20,6))
        for i, class_id in enumerate(selected_classes):
            class_dir = self.cfg.TRAIN_DIR / class_id
            images = list(class_dir.glob('*.png'))
            if not images: continue

            img_path = np.random.choice(images)
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cols = int(math.ceil(num_samples/2))
                plt.subplot(2, cols, i+1)
                plt.imshow(img)
                plt.title(self.label_map.get(int(class_id), class_id), fontsize=9)
                plt.axis('off')
        plt.tight_layout()
        plt.show()

class Visualizer:
    """Utilities for visualizing data and results."""

    @staticmethod
    def show_samples(dataset, num_img=9):
        """Displays a grid of images."""
        plt.figure(figsize=(10, 10))

        for images, labels in dataset.take(1):
            for i in range(min(num_img, images.shape[0])):
                ax = plt.subplot(3, 3, i + 1)
                img = images[i].numpy()
                
                # Rescale for disp. if neeeded
                
                if img.min() < 0:
                    img = (img - img.min()) / (img.max() - img.min())

                plt.imshow(img)
                if len(labels[i].shape) > 0 and labels[i].shape[0] > 1:
                    title_idx = np.argmax(labels[i])
                else:
                    title_idx = int(labels[i])

                plt.title(f"Class: {title_idx}")
                plt.axis("off")
        plt.show()

    @staticmethod
    def plot_history(history):
        """Plots accuracy and loss curves."""
        hist = history.history if hasattr(history, 'history') else history
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        axes[0].plot(hist.get('accuracy', []), label='Train')
        axes[0].plot(hist.get('val_accuracy', []), label='Val')
        axes[0].set_title('Accuracy')
        axes[0].legend()

        # Loss
        axes[1].plot(hist.get('loss', []), label='Train')
        axes[1].plot(hist.get('val_loss', []), label='Val')
        axes[1].set_title('Loss')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model, dataset, class_map=None):
        """Generates and plots a conf.  matrix."""
        print("Generating confusion matrix...")
        y_true = []
        y_pred = []

        for img_batch, label_batch in dataset:
            preds = model.predict(img_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            if label_batch.shape[-1] > 1:
                y_true.extend(np.argmax(label_batch.numpy(), axis=1))
            else:
                y_true.extend(label_batch.numpy())

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(25, 25))

        if class_map:
            labels = [class_map[i] for i in sorted(class_map.keys())]
        else:
            labels = "auto"

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 9})

        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.title('Confusion Matrix', fontsize=24)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.show()

class TrafficSignModel:
    def __init__(self, config, model_type="MobileNetV2"):
        self.cfg = config
        self.model_type = model_type
        self.model = None
        self.base_model = None

    def build(self):
        inputs = keras.Input(shape=self.cfg.INPUT_SHAPE)
        x = layers.RandomRotation(0.06)(inputs)
        x = layers.RandomTranslation(0.08, 0.08)(x)
        x = layers.RandomZoom(0.08)(x)

        # Selecting base model
        if self.model_type == "MobileNetV2":
            self.base_model = applications.MobileNetV2(
                input_shape=self.cfg.INPUT_SHAPE, include_top=False, weights='imagenet', pooling='avg'
            )
        elif self.model_type == "ResNet50V2":
            self.base_model = applications.ResNet50V2(
                input_shape=self.cfg.INPUT_SHAPE, include_top=False, weights='imagenet', pooling='avg'
            )
        elif self.model_type == "EfficientNetB0":
            self.base_model = applications.EfficientNetB0(
                input_shape=self.cfg.INPUT_SHAPE, include_top=False, weights='imagenet', pooling='avg'
            )

        self.base_model.trainable = False

        x = self.base_model(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.cfg.NUM_CLASSES, activation='softmax')(x)

        self.model = models.Model(inputs, outputs, name=f'{self.model_type}_GTSRB')
        self.compile_model(learning_rate=self.cfg.LEARNING_RATE)
        return self.model

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )

    def train(self, train_ds, val_ds):
        model_path = self.cfg.MODELS_DIR / f"{self.cfg.PROJECT_NAME}_{self.model_type}_best.keras"
        callbacks = [
            ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
        ]
        return self.model.fit(train_ds, validation_data=val_ds, epochs=self.cfg.EPOCHS, callbacks=callbacks)

    def fine_tune(self, train_ds, val_ds):
        print(f" Fine-Tuning on {self.model_type}")
        self.base_model.trainable = True

        for layer in self.base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        self.compile_model(learning_rate=1e-5)
        
        model_path = self.cfg.MODELS_DIR / f"{self.model_type}_finetuned.keras"
        callbacks = [
            ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
        return self.model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

class TrafficSignPredictor:
    """Wrapper to use the trained model for inference on files or raw data."""

    def __init__(self, model_path, config, data_manager):
        self.cfg = config
        self.label_map = data_manager.load_signnames_csv()
        self.dm = data_manager

        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

    def preprocess(self, img_rgb):
        """Standard preprocessing pipeline."""
        img_resized = cv2.resize(img_rgb, self.cfg.IMG_SIZE)
        img_batch = tf.expand_dims(img_resized, axis=0)
        img_batch = tf.cast(img_batch, tf.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

    def predict(self, img_input, is_filepath=True):
        if is_filepath:
            img = cv2.imread(img_input)
            if img is None: return "Error: Image not found", 0.0
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_input

        # pp
        img_tensor = self.preprocess(img_rgb)
        preds = self.model.predict(img_tensor, verbose=0)

        # Decoding label
        idx = np.argmax(preds)
        conf = float(np.max(preds))

        # Map Index
        class_id_str = self.dm.class_names[idx]
        true_id = int(class_id_str)

        label = self.label_map.get(true_id, f"Class {true_id}")
        return label, conf

def evaluate_full(model_path, name, test_ds, y_true):
    """Evaluation helper function."""
    print(f"\n--- Evaluating {name} ---")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Could not load {model_path}. Error: {e}")
        return None

    # Inf. time check
    start = time.time()
    y_pred_probs = model.predict(test_ds, verbose=1)
    end = time.time()

    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculating Metrics
    # Note: y_true needs to be aligned with y_pred. 
    
    if len(y_true) != len(y_pred):
        print("Warning: Label count mismatch. Skipping detailed metrics.")
        return None

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    avg_time = (end - start) / len(y_true) * 1000

    print(f"Accuracy: {acc:.4f}")
    print(f"Time per image: {avg_time:.2f} ms")

    return {
        "name": name,
        "y_pred": y_pred,
        "accuracy": acc,
        "f1": f1,
        "time": avg_time
    }

def launch_gradio_app(config, dm, model_path):
    if not GRADIO_AVAILABLE:
        print("Gradio not installed. Skipping web app launch.")
        return

    print(f"Launching Gradio app using model: {model_path}")
    if not os.path.exists(model_path):
        print("Model not found for Gradio.")
        return

    predictor = TrafficSignPredictor(model_path, config, dm)

    def classify_image(input_img):
        if input_img is None: return None
        label, conf = predictor.predict(input_img, is_filepath=False)
        return {label: conf}

    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(label="Upload Traffic Sign"),
        outputs=gr.Label(num_top_classes=1),
        title="Traffic Sign Recognizer",
        description="Upload an image of a German Traffic Sign to identify it."
    )
    interface.launch(debug=True, share=False)

def main():
    # init
    config = Config()
    dm = GTSRBDataManager(config)
    
    # data setup
    if not config.TRAIN_DIR.exists():
        dm.setup_data()
    if not config.TRAIN_DIR.exists():
        print("Training data not found. Exiting.")
        return

    train_ds, val_ds = dm.get_datasets()
    
    # vis.
    
    print("Viewing training samples...")
    Visualizer.show_samples(train_ds)

    # Training loop
    
    model_types = ["MobileNetV2", "ResNet50V2"]
    trained_results = {}

    for name in model_types:
        print(f"\n{'='*40}")
        print(f"PROCESSING MODEL: {name}")
        print(f"{'='*40}")
        
        save_path = config.MODELS_DIR / f"{name}_final.keras"

        if save_path.exists():
            print(f"Model {name} already exists at {save_path}. Skipping training.")
        else:
            # Build & Train
            ts_model = TrafficSignModel(config, model_type=name)
            ts_model.build()

            print(f"Phase 1: Initial training...")
            ts_model.train(train_ds, val_ds)

            print(f"Phase 2: Fine-tuning...")
            hist_fine = ts_model.fine_tune(train_ds, val_ds)

            trained_results[name] = {'history': hist_fine.history}

            # Saving model
            ts_model.model.save(save_path)
            print(f"Saved model to {save_path}")

            # Cleanup
            del ts_model
            tf.keras.backend.clear_session()
            gc.collect()

    # eval.
    test_ds = dm.get_test_dataset()
    if test_ds:
        print("\nSTARTING TEST SET EVALUATION...")
        
        # ex. true labels
        y_true = []
        for imgs, labels in test_ds:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)

        results = []
        for name in model_types:
            path = config.MODELS_DIR / f"{name}_final.keras"
            res = evaluate_full(path, name, test_ds, y_true)
            if res:
                results.append(res)

        # Plot Comparison
        if len(results) >= 2:
            results_df = pd.DataFrame([
                {"Model": r['name'], "Accuracy": r['accuracy'], "Inference Time (ms)": r['time']}
                for r in results
            ])
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results_df, x="Model", y="Accuracy", color="skyblue", ax=ax1, alpha=0.6)
            ax1.set_ylabel("Accuracy", color="blue")
            ax1.set_ylim(0.8, 1.0)
            
            ax2 = ax1.twinx()
            sns.lineplot(data=results_df, x="Model", y="Inference Time (ms)", color="red", marker="o", ax=ax2, linewidth=3)
            ax2.set_ylabel("Inference Time (ms)", color="red")
            plt.title("Trade-off: Accuracy vs. Speed")
            plt.show()

    # launching app
    if model_types:
        best_model_path = config.MODELS_DIR / f"{model_types[0]}_final.keras"
        print("\nWould you like to launch the Gradio App? (y/n)")
        # launch_gradio_app(config, dm, best_model_path) 

if __name__ == "__main__":
    main()