import unittest
import torch
import pandas as pd
import numpy as np
from mml.models.metrics import MetricTracker, Metrics

class MockModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn(x.size(0), 1).squeeze(1)

class MockDataLoader:
    def __init__(self, num_batches, batch_size, num_classes):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                'image': torch.randn(self.batch_size, 3, 224, 224),
                'label': torch.randint(0, self.num_classes, (self.batch_size,)).float()
            }

    def __len__(self):
        return self.num_batches

class TestMetricTracker(unittest.TestCase):
    def setUp(self):
        self.classes = [0, 1]
        self.from_logits = False
        self.metrics = ['accuracy_score', 'precision_score']
        self.kwargs = {
            'f1_score':{
                    'labels': self.classes,
                    'pos_label': 1,
                    'average': "binary",
                    'sample_weight': None,
                    'zero_division': "warn"
            },
            'recall_score':{
                    'labels': self.classes,
                    'pos_label': 1,
                    'average': "binary",
                    'sample_weight': None,
                    'zero_division': "warn"
            },
            'precision_score': {
                    'labels': self.classes,
                    'pos_label': 1,
                    'average': "binary",
                    'sample_weight': None,
                    'zero_division': "warn"            
            },
            'accuracy_score': {
                    'normalize': True, 
                    'sample_weight': None          
            }
        }
        self.tracker = MetricTracker(self.metrics, self.classes, self.from_logits, self.kwargs)
        self.tracker_logits = MetricTracker(self.metrics, self.classes, True, self.kwargs)

    def test_init(self):
        self.assertEqual(self.tracker.metrics, self.metrics)
        self.assertEqual(self.tracker.kwargs, self.kwargs)
        self.assertIsInstance(self.tracker.metrics_df, pd.DataFrame)
        self.assertIsInstance(self.tracker.epoch_df, pd.DataFrame)
        self.assertIsInstance(self.tracker.metrics_wrapper, Metrics)

    def test_update(self):
        #predictions = [0, 1, 0, 1]
        predictions = torch.tensor([0.6,  0.3, 0.8, 0.4])
        labels = torch.tensor([0, 1, 0, 1])
        self.tracker.update(predictions, labels)
        self.assertEqual(len(self.tracker.epoch_df), 4)
        np.testing.assert_array_equal(self.tracker.epoch_df['predictions'], predictions)
        np.testing.assert_array_equal(self.tracker.epoch_df['labels'], labels)

    def test_reset(self):
        self.tracker.update([0, 1], [1, 0])
        self.tracker.reset()
        self.assertTrue(self.tracker.epoch_df.empty)

    def test_calculate_metrics(self):
        self.tracker.update([0, 1, 0, 1], [0, 1, 1, 0])
        metrics = self.tracker.calculate_metrics()
        self.assertIn('accuracy_score', metrics)
        self.assertIn('precision_score', metrics)

    def test_log_epoch_metrics(self):
        self.tracker.update([0, 1, 0, 1], [0, 1, 1, 0])
        epoch_metrics = self.tracker.log_epoch_metrics(1)
        self.assertIn('accuracy_score', epoch_metrics)
        self.assertIn('precision_score', epoch_metrics)
        self.assertEqual(len(self.tracker.metrics_df), 1)
        self.assertTrue(self.tracker.epoch_df.empty)

    def test_result(self):
        self.tracker.update([0, 1, 0, 1], [0, 1, 1, 0])
        self.tracker.log_epoch_metrics(1)
        result = self.tracker.result()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_val_epoch_simulation(self):
        model = MockModel()
        dataloader = MockDataLoader(num_batches=5, batch_size=32, num_classes=2)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(3):
            val_loss = 0
            for data in dataloader:
                sample = data['image']
                label = data['label']
                output = model(sample)
                loss = criterion(output, label)
                val_loss += loss.item()
                self.tracker_logits.update(predictions=output, labels=label)
            epoch_loss = val_loss / len(dataloader)
            epoch_metrics = self.tracker_logits.log_epoch_metrics(epoch)

            self.assertIsInstance(epoch_loss, float)
            for metric in self.metrics:
                self.assertIn(metric, epoch_metrics)
                self.assertIsInstance(epoch_metrics[metric], float)
        final_result = self.tracker_logits.result()
        self.assertEqual(len(final_result), 3)  # 3 epochs
        self.assertEqual(list(final_result.columns), ['epoch'] + self.metrics)

if __name__ == '__main__':
    unittest.main()