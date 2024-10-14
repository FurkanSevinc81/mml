import torch
from tqdm import tqdm
from mml.models.metrics import MetricTracker
from typing import List, Any
import os 


class Trainer():
    def __init__(self, model, criterion, optimizer, 
                 train_metrics: List[str], val_metrics: List[str], 
                 modality, kwargs_metrics_train, kwargs_metrics_val,
                 classes, from_logits: bool, epochs: int, logger,
                 train_dataloader, val_dataloader, lkso_dataloader,
                 epoch_start: int=0, device=None, dtype=None):
        self.logger = logger
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # save the initial model state_dict, such that it can be loaded for CV 
        self.init_checkpoint_path = self.logger.save_model_checkpoint(
            self.model, self.optimizer, 'init_checkpoint', 0
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lkso_dataloader = lkso_dataloader
        self.modalitiy = modality
        self.classes = classes
        self.binary = len(self.classes) == 2
        self.kwargs_metrics_train = kwargs_metrics_train
        self.kwargs_metrics_val = kwargs_metrics_val

        self.train_metric_tracker = MetricTracker(train_metrics, len(classes), 
                                                  from_logits, self.kwargs_metrics_train)
        self.val_metric_tracker = MetricTracker(val_metrics, len(classes),
                                                from_logits, self.kwargs_metrics_val)
        self.epochs = epochs
        self.epoch_start = epoch_start

    def _perpare_labels(self, label):
        labels = label.clone()
        if self.binary:
            negative_class = self.classes[0]
            positive_class = self.classes[1]
            labels[labels == negative_class] = 0
            labels[labels == positive_class] = 1

        labels = labels.float().to(**self.factory_kwargs)  
        return labels
    
    def _reset_model_and_optim(self):
        self.model.resume_checkpoint(self.optimizer, path=self.init_checkpoint_path)
    
    def _train_epoch(self, epoch, fold=""):
        train_loss = 0
        self.model.train()
        
        for batch_idx, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            sample = data[self.modalitiy]
            sample = sample.to(**self.factory_kwargs)
            label = data['label']
            label = self._perpare_labels(label)

            output = self.model(sample)
            if output.dim() == 2 and output.size(1) == 1:
                output = output.squeeze(1)
            output = output.float()
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            #### LOGGING  ####
            log_msg = (f"Training Progress - Epoch [{epoch+1}/{self.epochs}], "
                       f"Batch [{batch_idx+1}/{len(self.train_dataloader)}], "
                       f"Current Loss: {loss:.4f}")
            self.logger.log(log_msg, verbosity=1)
            self.logger.log('Updating train metric tracker predictions', verbosity=2)
            self.train_metric_tracker.update(predictions=output, labels=label)
        self.logger.log_df_to_file(os.path.join('training', f'train_predictions{fold}.csv'),
                                   self.train_metric_tracker.epoch_df, mode='a',
                                   verbosity=2)
        epoch_loss = train_loss / len(self.train_dataloader)
        self.logger.log('Calculating train epoch metrics', verbosity=2)
        self.logger.log('Updating train epoch metrics', verbosity=2)
        epoch_metrics = self.train_metric_tracker.log_epoch_metrics(epoch)
        self.logger.log('Clearing train predictions', verbosity=2)
        return epoch_loss, epoch_metrics
    
    def _val_epoch(self, epoch, fold=""):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_dataloader):
                sample = data[self.modalitiy]
                sample = sample.to(**self.factory_kwargs)
                label = data['label']
                label = self._perpare_labels(label)

                output = self.model(sample)
                if output.dim() == 2 and output.size(1) == 1:
                    output = output.squeeze(1)
                output = output.float()
                loss = self.criterion(output, label)
                val_loss += loss.item()
                #### LOGGING  ####
                log_msg = (f"Validation - Epoch [{epoch+1}/{self.epochs}], "
                           f"Batch [{batch_idx+1}/{len(self.val_dataloader)}], "
                           f"Current Loss: {loss:.4f}")
                self.logger.log(log_msg, verbosity=1)
                self.logger.log('Updating val metric tracker predictions', verbosity=2)
                self.val_metric_tracker.update(predictions=output, labels=label)
            self.logger.log_df_to_file(os.path.join('validation',f'val_predictions{fold}.csv'),
                                       self.val_metric_tracker.epoch_df, mode='a',
                                       verbosity=2) # should include the fold if given
            epoch_loss = val_loss / len(self.val_dataloader)
            self.logger.log('Calculating val epoch metrics', verbosity=2)
            self.logger.log('Updating val epoch metrics', verbosity=2)
            epoch_metrics = self.val_metric_tracker.log_epoch_metrics(epoch)
            #self.logger.log_to_tb({'loss': epoch_loss, **epoch_metrics}, epoch, phase='val') #should include the fold
            self.logger.log('Clearing val predictions', verbosity=2)
        return epoch_loss, epoch_metrics

    def train(self):
        if self.lkso_dataloader is None:
            self._train(None)
        else:
            self._k_fold_train()
    
    def _train(self, fold_idx=None):
        train_losses = []
        val_losses = []

        if fold_idx is None:
            fold_idx = ""
        else:
            fold_idx = f'_{fold_idx}'

        for epoch in range(self.epoch_start, self.epochs, 1):
            train_loss, train_met =self._train_epoch(epoch, fold_idx)
            train_losses.append(train_loss)
            phase_str = f'fold{fold_idx}/train' if fold_idx is not None else 'train'
            self.logger.log_to_tb({'loss': train_loss, **train_met}, epoch, phase=phase_str)  
            val_loss, val_met = self._val_epoch(epoch, fold_idx)
            val_losses.append(val_loss)
            phase_str = f'fold_{fold_idx}/val' if fold_idx is not None else 'val'
            self.logger.log_to_tb({'loss': val_loss, **val_met}, epoch, phase=phase_str)  
            # quicksave
            self.logger.save_model_checkpoint(model=self.model, optimizer=self.optimizer, 
                                              name=f'model{fold_idx}', epoch=epoch) 
        val_metrics = self.val_metric_tracker.result()
        train_metrics = self.train_metric_tracker.result()
        self.logger.log_df_to_file(file_path=os.path.join('metrics', f'val_metrics{fold_idx}.csv'), 
                                   df=val_metrics, mode='a', verbosity=1) 
        self.logger.log_df_to_file(file_path=os.path.join('metrics', f'train_metrics{fold_idx}.csv'), 
                                   df=train_metrics, mode='a', verbosity=1)  
        self.logger.close_tb()
        self.logger.save_model(self.model, f'model_obj{fold_idx}')  
        return train_losses, train_metrics, val_losses, val_metrics
    
    def _k_fold_train(self):
        for fold_idx, folds in enumerate(self.lkso_dataloader):
            self.logger.log(f"Training fold {fold_idx + 1}", verbosity=1)
            self.logger.log(f"Reseting model", verbosity=1)
            self._reset_model_and_optim()
            self.logger.log(f"Reseting validation metric tracker", verbosity=1)
            self.val_metric_tracker.reset_states()
            self.logger.log(f"Reseting training metric tracker", verbosity=1)
            self.train_metric_tracker.reset_states()
            self.train_dataloader = folds.train_dataloader
            self.val_dataloader = folds.test_dataloader
            train_losses, train_metrics, val_losses, val_metrics = self._train(fold_idx)