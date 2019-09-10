DEVICE = 'CPU'  # CPU | GPU
GPUS = 1  # Number of GPUS

VERBOSE = 1

PROBLEM_TYPE = 'classification'  # classification | regression
DATASET_NAME = 'OASIS'  # Dataset's name: ADNI, OASIS

MODEL_TYPE = 'resnet'  # Type of model for evaluate: full, convolutional, vgg16, mobile, mobilev2, resnet, unet, custom
MODEL_PATH = None  # Path to models json file, if it doesn't exist: None
WEIGHT_PATH = None  # Path to models h5 weight file, if it doesn't exist: None
STRATEGY_TYPE = 'loo'  # Strategy type: kfold, holdout, loo (leaving one out)

REPETITIONS = 50 # Number of strategy repetitions
FOLDS = 10  # Number of folds in kfolds
HOLDOUT_TEST_SPLIT = 0.20  # Holdout split proportion for test in range [0,1]
VALIDATION_SPLIT = 0.01  # Holdout split proportion of train for validation in range [0,1]

EPOCHS = 75
BATCH = 8

DENSE_LAYERS_LIST = ['dense','dense']
DENSE_DIMENSION_LIST = [512,2]

CONV_LAYERS_LIST = ['conv','maxpool'] # conv or maxpool
CONV_DIMENSION_LIST = [(11,11),(5,5)]

CUSTOM_LAYERS_LIST =  ['flatten','dense']
CUSTOM_DIMENSION_LIST = [None, 4]

USE_SKLEARN = True
SKLEARN_MODEL = 'svm'

USE_DISTILLERY = False
DISTILLERY_MODEL_PATH = None  # Path to distillery models json file, if it doesn't exist: None
DISTILLERY_WEIGHT_PATH = None  # Path to distillery models h5 weight file, if it doesn't exist: None

LAST_LAYERS = 3  # Last layers to remove from pretrained models

USE_DROPOUT = True
DROPOUT_PROB = 0.15  # Dropout probability

ACTIVATION = 'softmax'
OPTIMIZER = 'adam'
METRICS = ['accuracy','categorical_accuracy']
LOSS = 'categorical_crossentropy'

# Early stopping
MIN_DELTA = 0.05
PATIENCE = 3

# Evaluation in new dataset
NEW_DATASET = False
NEW_DATASET_NAME = 'OASIS'

AUGMENTATION = False # True if data augmentation, else false
