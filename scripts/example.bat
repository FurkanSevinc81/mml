@echo off

set "PYTHON_SCRIPT=train.py"

set "NAME_RBF=RBF_SMALL_GSR"
set "CONFIG_PATH=..\config\default_conf.yaml"
set "VERBOSITY=2"
set "SAVE_DIR=..\logs"
set "KERNEL_FUNCTION_RBF=rbf"
set "TRANSFORMER_SIZE=small"
set "EMBEDDING=high"
set "CLASSIFICATION=cls"
set "SPLIT=0.5"
set "EPOCHS=30"
set "CSV_FILE=..\datasets\BioVid\PartA\samples.csv"
set "DATA_ROOT=..\datasets\BioVid\PartA"
set "LOG_CONFIG=..\config\log_config.yaml"
set "MOD_GSR=gsr"
set "MOD_ECG=ecg"
python "%PYTHON_SCRIPT%" ^
    -c "%CONFIG_PATH%" ^
    --verbosity "%VERBOSITY%" ^
    -s "%SAVE_DIR%" ^
    --kf "%KERNEL_FUNCTION_RBF%" ^
    --ts "%TRANSFORMER_SIZE%" ^
    --em "%EMBEDDING%" ^
    --cls "%CLASSIFICATION%" ^
    --split "%SPLIT%" ^
    --ep "%EPOCHS%" ^
    --csv "%CSV_FILE%" ^
    --dr "%DATA_ROOT%" ^
    -n "%NAME_RBF%" ^
    -l "%LOG_CONFIG%" ^
    --mod "%MOD_GSR%"

python "%PYTHON_SCRIPT%" ^
    -c "%CONFIG_PATH%" ^
    --verbosity "%VERBOSITY%" ^
    -s "%SAVE_DIR%" ^
    --kf "%KERNEL_FUNCTION_RBF%" ^
    --ts "%TRANSFORMER_SIZE%" ^
    --em "%EMBEDDING%" ^
    --cls "%CLASSIFICATION%" ^
    --split "%SPLIT%" ^
    --ep "%EPOCHS%" ^
    --csv "%CSV_FILE%" ^
    --dr "%DATA_ROOT%" ^
    -n "%NAME_RBF%" ^
    -l "%LOG_CONFIG%" ^
    --mod "%MOD_GSR%"
