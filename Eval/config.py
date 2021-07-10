BATCH_SIZE = 6
LR = 0.0001
DROPOUT = 0.4  # 0.4
WEIGHT_DECAY = 1e-6
CLASSIFIER_EPOCHS = 25  # 40 time before staged unfreezing. Just the classifier and any base bert layers
RESAMPLE_TRAIN_DATA = True
RESAMPLE_METHOD = "min"  # "sum", "avg", "min"
RESAMPLE_STRENGTH = 0.125  # used as an exponent. smaller values lead to less drastic resampling
STAGED_LAYERS = ["layer.0", "layer.11", "layer.1", "layer.10", "layer.2"]
STAGE_EPOCHS = 25
FINISHING_EPOCHS = 0  # time after all layers have been unfrozen
MAX_BATCHES = 999  # for debug. makes epochs run faster
TRAIN_FRAC = 0.9
NUM_EPOCHS = CLASSIFIER_EPOCHS + len(STAGED_LAYERS) * STAGE_EPOCHS + FINISHING_EPOCHS