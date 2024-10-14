from mml.utils import config
import argparse
from collections import namedtuple
import argparse

def split_type(value):
    try:
      fvalue = float(value)
      if 0.0 <= fvalue <= 1.0:
            return fvalue
    except ValueError:
      pass
    try:
      ivalue = int(value)
      return ivalue
    except ValueError:
      pass

    raise argparse.ArgumentTypeError(f"Invalid value: {value}. \
                                     It should be an integer or a float between 0.0 and 1.0.")

def list_of_ints(value):
    try:
      values = value.split(',')
      int_values = [int(v) for v in values]
      return int_values
    except ValueError:
      raise argparse.ArgumentTypeError(f"Invalid list of integers: '{value}'. It should be \
                                       a comma-separated list of integers.")

def list_of_strings(value):
    try:
      values = value.split(',')
      str_values = [str(v) for v in values]
      return str_values
    except ValueError:
      raise argparse.ArgumentTypeError(f"Invalid list of strings: '{value}'. It should be a \
                                       comma-separated list of strings.")



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training a kernel Transformer model on the \
                                   BioVid database')
    args.add_argument('-c', '--config', metavar='PATH', default=None, type=str,
                      help='config file path(default: None)')
    args.add_argument('-r', '--resume', metavar='PATH', default=None, type=str,
                      help='path to latest checkpoint')

    CustomArgs = namedtuple('CustomArgs', 'flags type help target metavar')
    options = [
        CustomArgs(['-n', '--name'], type=str, target='name', help='experiment name',
                   metavar='STR'),
        CustomArgs(['-ld', '--load'], type=str, target='load_model', help='path to model to be loaded',
                   metavar='PATH'),
        CustomArgs(['-s', '--save'], type=str, target='save_dir', help='path to save directory. this \
                   will create subfolders for models and logs if not existent ',  metavar='PATH'),
        CustomArgs(['--kf', '--kernel_function'], type=str, target='model;kernel_function;name',
                   help='possible kernel functions: `linear`, `polynomial`, `rbf`, `sigmoid`, \
                        `laplacian`, `exponential`',  metavar=''),
        CustomArgs(['--ts', '--transformer_size'], type=str, target='model;kernel_transformer',
                   help='choose from `base`, `small`, `medium`, or `large`, or provide path to\
                        custom transformer config.',  metavar=''),
        CustomArgs(['--em', '--embedding'], type=str, target='model;embedding', 
                   help='',  metavar=''),
        CustomArgs(['--cls', '--classification'], type=str, target='model;classification;mode', 
                   help='',  metavar=''),
        CustomArgs(['--mod', '--modality'], type=str, target='trainer;modality', 
                   help='', metavar=''),
        CustomArgs(['--split'], type=split_type, target='data;test_size', help='Test split: an \
                   integer or a float between 0.0 and 1.0', metavar=''),     
        CustomArgs(['--classes'], type=list_of_ints, target='data;classes', help='List of class \
                   indices: a comma-separated list of integers, e.g., "0,4,7"', metavar=''), 
        CustomArgs(['--bs', '--batch_size'], type=int, target='data;batch_size', 
                   help='', metavar=''),  
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs', 
                   help='', metavar=''),
        CustomArgs(['--tm', '--train_metrics'], type=list_of_strings, target='trainer;train_metrics', 
                   help='', metavar=''),
        CustomArgs(['--vm', '--val_metrics'], type=list_of_strings, target='trainer;val_metrics', 
                   help='', metavar=''),
        CustomArgs(['--csv', '--csv_file'], type=str, target='data;csv_file', 
                   help='', metavar='PATH'),
        CustomArgs(['--dr', '--data_root'], type=str, target='data;root_dir', 
                   help='', metavar='PATH'),
        CustomArgs(['--dev', '--device'], type=str, target='factory_kwargs;device', 
                   help='', metavar='DEVICE'),
        CustomArgs(['--dt', '--dtype'], type=str, target='factory_kwargs;dtype', 
                   help='', metavar='DTYPE'),
        CustomArgs(['-l', '--log_config'], type=str, target='log_config', 
                   help='log config file path(default: None)', metavar='PATH'),
        CustomArgs(['-v', '--verbosity'], type=int, target='verbosity', help='Set the verbosity\
                    level: 0=WARNING, 1=INFO, 2=DEBUG', metavar='N'),
        CustomArgs(['--id', '--run_id'], type=str, target='run_id', help='ID of the run. If not\
                   specified the current date will be used.', 
                   metavar='ID'),
        CustomArgs(['--cv_k'], type=int, target='data;k', help='Number of folds\
                   for leave-k-subjects-out cross-validation.', 
                   metavar='N')
    ]

    train_conf = config.ConfigParser.from_args(args, options)
    trainer = train_conf.init_trainer()
    trainer.train()