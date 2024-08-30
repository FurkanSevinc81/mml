import torch
from torch import Tensor
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from typing import List, Union, Any, Dict

metric_mapping = {
    'accuracy': 'accuracy_score',
    'f1': 'f1_score',
    'precision': 'precision_score',
    'recall': 'recall_score',
    'roc_auc': 'roc_auc_score',
    'top_k': 'top_k_accuracy_score'
     }

class Metrics():
    def __init__(self, num_classes: List[Any], from_logits: bool):
        self.multi_class = num_classes != 2
        self.from_logits = from_logits
    
    def __call__(self, metric: str, output: Union[Tensor, np.ndarray], 
                 target: Union[Tensor, np.ndarray], **kwargs) -> Any:
        y_pred, y_true = self._prepare_output_target(output, target)
        metric_name = metric_mapping.get(metric, metric)

        if not hasattr(skmetrics, metric_name):
            raise ValueError(f'Metric {metric} not found.')
        metric_function = getattr(skmetrics, metric_name)

        return metric_function(y_true, y_pred, **kwargs)
    
    def _prepare_output_target(self, output, target):
        with torch.no_grad():
            if isinstance(target, Tensor):
                target = target.int().cpu().numpy()
            if not isinstance(output, Tensor):
                    output = torch.tensor(output)
            if self.from_logits: 
                if self.multi_class:
                    output = torch.softmax(output, dim=1) 
                else:
                    output = torch.sigmoid(output)
            if self.multi_class:
                output = output.argmax(dim=1)
            else:
                output = (output > 0.5).float()
            if isinstance(output, Tensor):
                output = output.int().cpu().numpy()
        return output, target
    
class MetricTracker():
    def __init__(self, metrics, num_classes, from_logits, kwargs):
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.kwargs = kwargs
        self.metrics_df = pd.DataFrame(columns=['epoch', *metrics])
        self.epoch_df = pd.DataFrame(columns=['predictions', 'labels'])

        self.metrics_wrapper = Metrics(num_classes, from_logits)

    def calculate_metrics(self):
        metrics = {}
        for metric in self.metrics:
            kwargs_metric = self.kwargs.get(metric, {})
            metric_score = self.metrics_wrapper(metric, 
                                                self.epoch_df['predictions'].tolist(), 
                                                self.epoch_df['labels'].tolist(), 
                                                **kwargs_metric)
            metrics[metric] = metric_score
        return metrics
    
    def update(self, predictions, labels):
        if isinstance(predictions, Tensor):
            predictions = predictions.detach()
        new_data = pd.DataFrame({'predictions': predictions, 'labels': labels})
        if self.epoch_df.empty:
            self.epoch_df = new_data
        else:
            self.epoch_df = pd.concat([self.epoch_df, new_data], ignore_index=True)

    def update_state(self, epoch, metrics):
        data = {'epoch': epoch, **metrics}
        new_data = pd.DataFrame([data])
        if self.metrics_df.empty:
            self.metrics_df = new_data
        else:
            self.metrics_df = pd.concat([self.metrics_df, new_data], ignore_index=True)

    def reset(self):
        self.epoch_df = pd.DataFrame(columns=['predictions', 'labels'])

    def reset_states(self):
        self.metrics_df = pd.DataFrame(columns=['epoch', *self.metrics])

    def log_epoch_metrics(self, epoch):
        epoch_metrics = self.calculate_metrics()
        self.update_state(epoch, epoch_metrics)
        self.reset()
        return epoch_metrics
    
    def result(self):
        return self.metrics_df