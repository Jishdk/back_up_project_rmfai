# main.py
import logging
from pathlib import Path
from config import OUTPUT_DIR, RESULTS_DIR
from preprocessing import DatasetPreprocessor
from utils import setup_logging
from model import BaselineModel
from evaluate import ModelEvaluator

logger = setup_logging()

def run_preprocessing():
    """Run preprocessing pipeline"""
    try:
        logger.info("Initializing data preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # Process TrashNet dataset
        logger.info("Processing TrashNet dataset...")
        preprocessor.process_trashnet()
        preprocessor.create_cross_validation_folds('trashnet')
        preprocessor.log_dataset_stats('trashnet')
        preprocessor.save_dataset_metadata('trashnet')
        
        # Process TACO dataset
        logger.info("Processing TACO dataset...")
        preprocessor.process_taco()
        preprocessor.create_cross_validation_folds('taco')
        preprocessor.log_dataset_stats('taco')
        preprocessor.save_dataset_metadata('taco')
        
        logger.info(f"Preprocessing complete. Output saved to {OUTPUT_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def run_baseline_evaluation():
    """Run baseline model evaluation"""
    try:
        logger.info("Starting baseline model evaluation...")
        model = BaselineModel()
        evaluator = ModelEvaluator()
        
        # Evaluate on both datasets
        for dataset in ['trashnet', 'taco']:
            logger.info(f"Evaluating baseline model on {dataset}...")
            data_yaml = OUTPUT_DIR / dataset / "dataset.yaml"
            try:
                results = model.predict(data_yaml)
                metrics = evaluator.evaluate_baseline(results, dataset)
                logger.info(f"{dataset} baseline mAP50: {metrics.get('mAP50', 'N/A')}")
            except Exception as e:
                logger.error(f"Error evaluating {dataset}: {e}")
        
        logger.info(f"Baseline evaluation complete. Results saved to {RESULTS_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error during baseline evaluation: {str(e)}")
        return False

def main():
    """Main execution pipeline"""
    try:
        # Run preprocessing if needed
        if not (OUTPUT_DIR / "trashnet" / "dataset.yaml").exists():
            if not run_preprocessing():
                raise RuntimeError("Preprocessing failed")
        
        # Run baseline evaluation
        if not run_baseline_evaluation():
            raise RuntimeError("Baseline evaluation failed")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()