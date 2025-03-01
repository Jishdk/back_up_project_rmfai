import os
from pathlib import Path

# Get directory containing the script
CODE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to code directory
DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = CODE_DIR / "preprocessed_data"
RESULTS_DIR = CODE_DIR / "results"  # Changed from OUTPUT_DIR/results to CODE_DIR/results

# Create specific results directories
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
TRAINED_RESULTS_DIR = RESULTS_DIR / "trained"
AUGMENTED_RESULTS_DIR = RESULTS_DIR / "augmented"

# Dataset directories
TRASHNET_DIR = DATA_DIR / "data_trashnet"  
TACO_DIR = DATA_DIR / "data_taco"          

# Image processing
IMG_SIZE = 640  # YOLOv8 optimal size
COLOR_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
COLOR_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Dataset split ratios (for TrashNet)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation settings
AUGMENTATION_ENABLED = True  # Control whether to use augmentation
AUGMENTATION_FACTOR = 2     # How many augmented versions per image

# Cross validation
CV_FOLDS = 3
RANDOM_STATE = 42

# Classes definitions
TRASHNET_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
TACO_CLASSES = [
    'Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can',
    'Carton', 'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic',
    'Paper', 'Plastic bag - wrapper', 'Plastic container', 'Pop tab',
    'Straw', 'Styrofoam piece', 'Unlabeled litter'
]

# Categories mapping (TACO to TrashNet)
CATEGORY_MAPPING = {
    'Bottle': 'plastic',
    'Plastic container': 'plastic',
    'Plastic bag - wrapper': 'plastic',
    'Other plastic': 'plastic',
    'Can': 'metal',
    'Bottle cap': 'metal',
    'Pop tab': 'metal',
    'Aluminium foil': 'metal',
    'Broken glass': 'glass',
    'Paper': 'paper',
    'Carton': 'cardboard',
    'Cup': 'trash',
    'Lid': 'trash',
    'Other litter': 'trash',
    'Cigarette': 'trash',
    'Straw': 'trash',
    'Styrofoam piece': 'trash',
    'Unlabeled litter': 'trash'
}

# YOLOv8 model configurations
YOLO_MODELS = {
    'n': 'nano',
    's': 'small',
    'm': 'medium',
    'l': 'large',
    'x': 'xlarge'
}
DEFAULT_MODEL = 'n'  # Default to nano model for baseline

# Evaluation Settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
METRICS = [
    'mAP50',        # mean Average Precision at IoU=0.50
    'mAP50-95',     # mean Average Precision at IoU=0.50:0.95
    'precision',    # Precision score
    'recall'        # Recall score
]

# Visualization Settings
FIGURE_SIZES = {
    'confusion_matrix': (12, 10),
    'class_performance': (12, 6)
}

# Create required directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BASELINE_RESULTS_DIR, exist_ok=True)
os.makedirs(TRAINED_RESULTS_DIR, exist_ok=True)
os.makedirs(AUGMENTED_RESULTS_DIR, exist_ok=True)

# Create dataset-specific baseline results directories
os.makedirs(BASELINE_RESULTS_DIR / "trashnet", exist_ok=True)
os.makedirs(BASELINE_RESULTS_DIR / "taco", exist_ok=True)