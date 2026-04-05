import json
import os
from datetime import datetime

class ModelMetrics:
    def __init__(self, metrics_file='model_metrics.json'):
        self.metrics_file = metrics_file
        self.metrics = self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
        
        # Return default structure if file doesn't exist
        return {
            'classification': {},
            'regression': {},
            'gan': {},
            'robot': {},
            'last_updated': None
        }
    
    def update_classification_metrics(self, precision, recall, f1, accuracy, loss, epochs, dataset_size):
        """Update classification metrics"""
        self.metrics['classification'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'loss': loss,
            'epochs': epochs,
            'datasetSize': f"{dataset_size:,} images",
            'last_trained': datetime.now().isoformat()
        }
        self.save_metrics()
    
    def update_regression_metrics(self, mse, mae, r2, rmse, epochs, dataset_size):
        """Update regression metrics"""
        self.metrics['regression'] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'epochs': epochs,
            'datasetSize': f"{dataset_size:,} samples",
            'last_trained': datetime.now().isoformat()
        }
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            self.metrics['last_updated'] = datetime.now().isoformat()
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")

# Global instance
metrics_manager = ModelMetrics()