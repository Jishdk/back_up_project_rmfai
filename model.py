# model.py
from pathlib import Path
import logging
from ultralytics import YOLO
from config import *
from utils import setup_logging

logger = setup_logging()

class BaselineModel:
    """Baseline YOLO model for garbage detection (RQ1)
    
    This class implements the out-of-the-box YOLOv8 model without any training.
    It will be used to establish a baseline performance as specified in RQ1.
    """
    
    def __init__(self, model_size: str = 'n'):
        """Initialize baseline model with pretrained weights
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
                n: nano (fastest, least accurate)
                s: small
                m: medium
                l: large
                x: extra large (slowest, most accurate)
        """
        # Load pretrained YOLOv8 model (no training needed for baseline)
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading pretrained YOLOv8 model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, data_yaml: Path, split: str = 'test') -> object:
        """Run predictions using the baseline model
        
        Args:
            data_yaml: Path to dataset YAML configuration file
            split: Dataset split to use ('train', 'val', 'test')
            
        Returns:
            YOLO Results object containing predictions and metrics
        """
        try:
            # Determine dataset name from data_yaml path
            dataset_name = data_yaml.parent.name  # Will be 'trashnet' or 'taco'
            results_dir = BASELINE_RESULTS_DIR / dataset_name
            
            logger.info(f"Running predictions on {dataset_name} {split} set")
            
            # Run validation without any training (baseline performance)
            results = self.model.val(
                data=str(data_yaml),
                split=split,
                imgsz=IMG_SIZE,
                batch=16,
                save_txt=True,
                save_conf=True,
                project=str(BASELINE_RESULTS_DIR),  # Save to baseline results directory
                name=dataset_name,  # Create dataset-specific subdirectory
                exist_ok=True  # Overwrite existing results
            )
            
            logger.info(f"Predictions completed for {dataset_name} {split} set")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise