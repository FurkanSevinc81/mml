#!/bin/bash

PYTHON_SCRIPT="train.py"

NAME_RBF="RBF_SMALL_GSR"
NAME_RBF="RBF_SMALL_ECG"
NAME_RBF="RBF_SMALL_EMG"
CONFIG_PATH="../config/default_conf.yaml"
VERBOSITY=2
SAVE_DIR="../logs"
KERNEL_FUNCTION_RBF="rbf"
TRANSFORMER_SIZE="small"
EMBEDDING="high"
CLASSIFICATION="cls"
SPLIT=0.5
EPOCHS=30
CSV_FILE="../datasets/BioVid/PartA/samples.csv"
DATA_ROOT="../datasets/BioVid/PartA"
LOG_CONFIG="../config/log_config.yaml"
MOD_GSR="gsr"
MOD_ECG="ecg"
MOD_EMG="emg_trapezius"

python "$PYTHON_SCRIPT" \
    -c "$CONFIG_PATH" \
    --verbosity "$VERBOSITY" \
    -s "$SAVE_DIR" \
    --kf "$KERNEL_FUNCTION_RBF" \
    --ts "$TRANSFORMER_SIZE" \
    --em "$EMBEDDING" \
    --cls "$CLASSIFICATION" \
    --split "$SPLIT" \
    --ep "$EPOCHS" \
    --csv "$CSV_FILE" \
    --dr "$DATA_ROOT" \
    -n "$NAME_RBF_GSR" \
    -l "$LOG_CONFIG" \
    --mod "$MOD_GSR"

python "$PYTHON_SCRIPT" \
    -c "$CONFIG_PATH" \
    --verbosity "$VERBOSITY" \
    -s "$SAVE_DIR" \
    --kf "$KERNEL_FUNCTION_RBF" \
    --ts "$TRANSFORMER_SIZE" \
    --em "$EMBEDDING" \
    --cls "$CLASSIFICATION" \
    --split "$SPLIT" \
    --ep "$EPOCHS" \
    --csv "$CSV_FILE" \
    --dr "$DATA_ROOT" \
    -n "$NAME_RBF_ECG" \
    -l "$LOG_CONFIG" \
    --mod "$MOD_ECG"

python "$PYTHON_SCRIPT" \
    -c "$CONFIG_PATH" \
    --verbosity "$VERBOSITY" \
    -s "$SAVE_DIR" \
    --kf "$KERNEL_FUNCTION_RBF" \
    --ts "$TRANSFORMER_SIZE" \
    --em "$EMBEDDING" \
    --cls "$CLASSIFICATION" \
    --split "$SPLIT" \
    --ep "$EPOCHS" \
    --csv "$CSV_FILE" \
    --dr "$DATA_ROOT" \
    -n "$NAME_RBF_EMG" \
    -l "$LOG_CONFIG" \
    --mod "$MOD_EMG"