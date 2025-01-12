# evaluate.py
import logging
from pathlib import Path
import json
from typing import Dict
from config import BASELINE_RESULTS_DIR, TRAINED_RESULTS_DIR, AUGMENTED_RESULTS_DIR, METRICS

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for garbage detection models"""
    
    def __init__(self):
        """Initialize model evaluator"""
        # Ensure all results directories exist
        BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        TRAINED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        AUGMENTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
    def evaluate_baseline(self, results: object, dataset_name: str) -> Dict:
        """Extract basic metrics from baseline evaluation
        
        Args:
            results: YOLO Results object from model validation
            dataset_name: Name of dataset ('trashnet' or 'taco')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract basic metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            # Create dataset-specific directory
            dataset_dir = BASELINE_RESULTS_DIR / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics to JSON
            metrics_file = dataset_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"Saved {dataset_name} baseline metrics to {metrics_file}")
            
            # Save confusion matrix if available
            if hasattr(results, 'confusion_matrix'):
                matrix_file = dataset_dir / "confusion_matrix.json"
                matrix_data = {
                    'matrix': results.confusion_matrix.matrix.tolist(),
                    'names': results.names
                }
                with open(matrix_file, 'w') as f:
                    json.dump(matrix_data, f, indent=4)
                logger.info(f"Saved confusion matrix to {matrix_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name} baseline: {e}")
            return {'error': str(e)}
            
    def get_metrics_summary(self, dataset_name: str) -> Dict:
        """Get summary of all metrics for a dataset
        
        Args:
            dataset_name: Name of dataset ('trashnet' or 'taco')
            
        Returns:
            Dictionary containing all available metrics
        """
        metrics_file = BASELINE_RESULTS_DIR / dataset_name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}