import logging
import logging.config
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

""" Functionality

    Metric Logging

    Error Handling

    Model Checkpointing

    Data Logging

    Time Tracking

    Custom Tags and Annotations

    Integration with External Tools

    Visualization Support

    Final Summary

    Archiving and Cleanup

    Reproducability

    Advanced Features
"""
class Logger:
    def __init__(self, save_dir_model, save_dir_logs, log_config, verbosity=2, 
                 default_level=logging.INFO):
        self.save_dir_model = save_dir_model
        self.save_dir_logs = save_dir_logs
        self.tb_log_dir = os.path.join(self.save_dir_logs, 'tb_logs')
        self.log_config = log_config
        self.default_level = default_level
        self._logger = None
        self._tb_writer = None

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        msg_verbosity = 'verbosity option {} is invalid. Valid options \
            are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        self.file_log_level = self.log_levels[verbosity]

        self._setup_logging()
        self._setup_tb_writer()

    def _setup_logging(self):
        if os.path.exists(self.log_config):
            with open(self.log_config, 'r') as file:
                log_config = yaml.safe_load(file)  

            for name, handler in log_config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = os.path.join(self.save_dir_logs,
                                                       handler['filename'])
                if name == 'info_file_handler':
                    handler['level'] = self.file_log_level
            logging.config.dictConfig(log_config)
            self._logger = logging.getLogger()
            if self._logger.level > self.file_log_level:
                self._logger.setLevel(self.file_log_level)
        else:
            print(f"Warning: logging configuration file is not found in {self.log_config}.")
            logging.basicConfig(level=self.default_level)
            self._logger = logging.getLogger()

    def _setup_tb_writer(self): 
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=self.tb_log_dir)

    def log(self, msg, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options \
            are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        level = self.log_levels[verbosity]
        if self._logger:
            self._logger.log(level, msg)

    def log_to_tb(self, metrics, epoch, phase='train'):
        self.log("Logging metrics to TensorBoard", verbosity=2)
        if self._tb_writer is not None:
            for metric_name, value in metrics.items():
                tag = f'{phase}/{metric_name}'
                self._tb_writer.add_scalar(tag, value, epoch)
            self._tb_writer.flush()
        else:
            self.log("TensorBoard writer not initialized.", verbosity=2)

    def log_df_to_file(self, file_path, df:pd.DataFrame, mode='a', verbosity=2):
        save_file = os.path.join(self.save_dir_logs, file_path)
        save_dir = os.path.dirname(save_file)
        os.makedirs(save_dir, exist_ok=True)
        self.log(f'Writing pandas dataframe to file {save_file} (mode:{mode})',
                 verbosity=verbosity)
        df.to_csv(save_file, mode=mode, index=False)

    def save_config(self, config_dict, save_dir):
        with open(save_dir, 'w') as save_file:
            yaml.dump(config_dict, save_file, default_flow_style=False)

    def save_model(self, model, name):
        self.log(f'Saving model object to {self.save_dir_model}', verbosity=2)
        model.save_model(name=f'{name}.pth', path=self.save_dir_model)

    def save_model_checkpoint(self, model, optimizer, name, epoch):
        self.log(f'Saving model checkpoint to {self.save_dir_model}', verbosity=2)
        return model.save_checkpoint(optimizer=optimizer, epoch=epoch, name=f'{name}_states.pt',
                              path=self.save_dir_model)

    def close_tb(self):
        if self._tb_writer:
            self._tb_writer.close()