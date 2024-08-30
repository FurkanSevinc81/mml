import unittest
import torch
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
from mml.trainer import Trainer

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        # 
        return (x + 0 * self.dummy_param).unsqueeze(1)
    
class MockDataLoader:
    def __init__(self, num_batches, batch_size, num_classes):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                #'image': torch.randn(self.batch_size, 3, 224, 224),
                'label': torch.randint(0, self.num_classes, (self.batch_size,)).float(),
                'pred': torch.rand((self.batch_size,)).float()
            }

    def __len__(self):
        return self.num_batches

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.classes = [0, 1]
        self.from_logits = True
        self.metrics_train = ['accuracy']
        self.metrics_val = ['accuracy', 'precision', 'recall', 'f1']
        self.kwargs_metrics_train = {
            'accuracy_score': {
                'normalize': True,
                'sample_weight': None          
            }
        }
        self.kwargs_metrics_val = {
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
   
        #self.labels = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0])
        self.modality = 'pred'
        self.num_batches = 4
        self.batch_size = 10
        self.epochs = 2
        self.model = MockModel()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.dataloader = MockDataLoader(self.num_batches, self.batch_size, 
                                         len(self.classes))

        self.trainer = Trainer(  
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            train_metrics=self.metrics_train,
            val_metrics=self.metrics_val,
            train_dataloader=self.dataloader,
            val_dataloader=self.dataloader,
            modality=self.modality,  
            classes=self.classes,
            from_logits=self.from_logits,
            kwargs_metrics_train=self.kwargs_metrics_train,
            kwargs_metrics_val=self.kwargs_metrics_val,
            epochs=self.epochs,
            device=self.device
        )

    def test_trainer_with_fixed_data(self):
        train_losses, train_metrics, val_losses, val_metrics = self.trainer.train()

        # Calculate expected metrics using sklearn
        y_true = self.labels.numpy()
        y_pred = self.predictions.argmax(dim=1).numpy()
        y_pred_proba = self.predictions[:, 1].numpy()  # Probability of positive class

        expected_metrics = {
            'accuracy': sk_metrics.accuracy_score(y_true, y_pred),
            'precision': sk_metrics.precision_score(y_true, y_pred, average='binary'),
            'recall': sk_metrics.recall_score(y_true, y_pred, average='binary'),
            'f1': sk_metrics.f1_score(y_true, y_pred, average='binary')
        }

        # Compare calculated metrics with expected metrics
        for metric in self.metrics:
            calculated = val_metrics[metric].iloc[-1]  # Get the last epoch's metric
            expected = expected_metrics[metric]
            self.assertAlmostEqual(calculated, expected, places=5,
                                   msg=f"Mismatch in {metric}. "
                                       f"Calculated: {calculated}, Expected: {expected}")

        # Additional checks
        self.assertEqual(len(train_losses), 1)  # One epoch
        self.assertEqual(len(val_losses), 1)  # One epoch
        self.assertEqual(len(train_metrics), 1)  # One epoch
        self.assertEqual(len(val_metrics), 1)  # One epoch

if __name__ == '__main__':
    unittest.main()