import unittest
import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from mml.models.metrics import Metrics 

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.binary_labels = [0, 1]
        self.multi_labels = [0, 1, 2]

    def test_binary_classification_from_logits(self):
        metrics = Metrics(labels=self.binary_labels, from_logits=True)
        
        logits = torch.randn(100, 1)
        targets = torch.randint(0, 2, (100,))

        preds = (torch.sigmoid(logits) > 0.5).int().squeeze()

        self._test_metric(metrics, 'f1_score', logits, targets, preds, f1_score)
        self._test_metric(metrics, 'recall_score', logits, targets, preds, recall_score)
        self._test_metric(metrics, 'precision_score', logits, targets, preds, precision_score)
        self._test_metric(metrics, 'accuracy_score', logits, targets, preds, accuracy_score)

    def test_binary_classification_without_logits(self):
        metrics = Metrics(labels=self.binary_labels, from_logits=False)

        preds = torch.randint(0, 2, (100,))
        targets = torch.randint(0, 2, (100,))

        self._test_metric(metrics, 'f1_score', preds, targets, preds, f1_score)
        self._test_metric(metrics, 'recall_score', preds, targets, preds, recall_score)
        self._test_metric(metrics, 'precision_score', preds, targets, preds, precision_score)
        self._test_metric(metrics, 'accuracy_score', preds, targets, preds, accuracy_score)

    def test_multiclass_classification_from_logits(self):
        metrics = Metrics(labels=self.multi_labels, from_logits=True)

        logits = torch.randn(100, 3)
        targets = torch.randint(0, 3, (100,))

        preds = torch.argmax(logits, dim=1)

        self._test_metric(metrics, 'f1_score', logits, targets, preds, f1_score, average='macro')
        self._test_metric(metrics, 'recall_score', logits, targets, preds, recall_score, average='macro')
        self._test_metric(metrics, 'precision_score', logits, targets, preds, precision_score, average='macro')
        self._test_metric(metrics, 'accuracy_score', logits, targets, preds, accuracy_score)

    def test_multiclass_classification_without_logits(self):
        metrics = Metrics(labels=self.multi_labels, from_logits=False)

        preds = torch.randint(0, 3, (100,))
        targets = torch.randint(0, 3, (100,))

        self._test_metric(metrics, 'f1_score', preds, targets, preds, f1_score, average='macro')
        self._test_metric(metrics, 'recall_score', preds, targets, preds, recall_score, average='macro')
        self._test_metric(metrics, 'precision_score', preds, targets, preds, precision_score, average='macro')
        self._test_metric(metrics, 'accuracy_score', preds, targets, preds, accuracy_score)

    def _test_metric(self, metrics, metric_name, output, target, preds, sklearn_func, **kwargs):
        result = metrics(metric_name, output, target, **kwargs)

        sklearn_result = sklearn_func(target.numpy(), preds.numpy(), **kwargs)

        self.assertAlmostEqual(result, sklearn_result, places=5)

if __name__ == '__main__':
    unittest.main()