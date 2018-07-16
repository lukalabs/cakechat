import os

from cakechat.utils.data_structures import create_namedtuple_instance
from cakechat.utils.env import is_dev_env

RANDOM_SEED = 42  # Fix the random seed to a certain value to make everything reproducible

# AWS S3 params
S3_MODELS_BUCKET_NAME = 'cake-chat-data'  # S3 bucket with all the data
S3_NN_MODEL_REMOTE_DIR = 'nn_models'  # S3 remote directory with models itself
S3_TOKENS_IDX_REMOTE_DIR = 'tokens_index'  # S3 remote directory with tokens index
S3_CONDITIONS_IDX_REMOTE_DIR = 'conditions_index'  # S3 remote directory with conditions index
S3_W2V_REMOTE_DIR = 'w2v_models'  # S3 remote directory with pre-trained w2v models

# data params
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')  # Directory to store all the data
# e.g. datasets, models, indices
NN_MODELS_DIR = os.path.join(DATA_DIR, 'nn_models')  # Path to a directory for saving and restoring dialog models
PROCESSED_CORPUS_DIR = os.path.join(DATA_DIR, 'corpora_processed')  # Path to a processed corpora datasets
TOKEN_INDEX_DIR = os.path.join(DATA_DIR, 'tokens_index')  # Path to a prepared tokens index file
CONDITION_IDS_INDEX_DIR = os.path.join(DATA_DIR, 'conditions_index')  # Path to a prepared conditions index file

# train & val data params
BASE_CORPUS_NAME = 'processed_dialogs'  # Basic corpus name prefix
TRAIN_CORPUS_NAME = 'train_' + BASE_CORPUS_NAME  # Corpus name prefix for the training dataset
CONTEXT_SENSITIVE_VAL_CORPUS_NAME = 'val_' + BASE_CORPUS_NAME  # Corpus name prefix for the validation dataset

MAX_VAL_LINES_NUM = 10000  # Max lines number from validation set to be used for metrics calculation
VAL_SUBSET_SIZE = 250  # Subset from the validation dataset to be used to calculated some validation metrics
TRAIN_SUBSET_SIZE = int(os.environ['SLICE_TRAINSET']) if 'SLICE_TRAINSET' in os.environ else None  # Subset from the
# training dataset to be used during the training. In case of None use all lines in the train dataset (default behavior)

# test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'quality')
CONTEXT_FREE_VAL_CORPUS_NAME = 'context_free_validation_set'  # Context-free validation set path
TEST_CORPUS_NAME = 'context_free_test_set'  # Context-free test set path
QUESTIONS_CORPUS_NAME = 'context_free_questions'  # Context-free questions only path

# word embeddings params
USE_PRETRAINED_W2V_EMBEDDINGS_LAYER = True  # Whether to use word2vec to pre-train weights for the embedding layer
TRAIN_WORD_EMBEDDINGS_LAYER = True  # Allow fine-tuning of the word embedding layer during the model training
W2V_MODEL_DIR = os.path.join(DATA_DIR, 'w2v_models')  # Path to store & load trained word2vec models
WORD_EMBEDDING_DIMENSION = 128  # word2vec embedding dimension
W2V_WINDOW_SIZE = 10  # word2vec window size, used during the w2v pre-training
USE_SKIP_GRAM = True  # Use skip-gram word2vec mode. When False, CBOW is used
MIN_WORD_FREQ = 1  # Minimum frequency of a word to be used in word2vec pre-calculation

# condition inputs. We use five major emotions to condition our model's predictions
EMOTIONS_TYPES = create_namedtuple_instance(
    'EMOTIONS_TYPES', neutral='neutral', anger='anger', joy='joy', fear='fear', sadness='sadness')
DEFAULT_CONDITION = EMOTIONS_TYPES.neutral  # Default condition to be used during the prediction (if not specified)
CONDITION_EMBEDDING_DIMENSION = 128  # Conditions embedding layer dimension to be trained.

# NN architecture params
ENCODER_DEPTH = 2  # Number of recurrent (GRU) layers for the encoder
DECODER_DEPTH = 2  # Number of recurrent (GRU) layers for the decoder
HIDDEN_LAYER_DIMENSION = 512  # Dimension for the recurrent layer
DENSE_DROPOUT_RATIO = 0.2  # Use dropout with the given ratio before decoder's output

# training params
INPUT_SEQUENCE_LENGTH = 30  # Input sequence length for the model during the training;
INPUT_CONTEXT_SIZE = 3  # Maximum depth of the conversational history to be used in encoder (at least 1)
OUTPUT_SEQUENCE_LENGTH = 32  # Output sequence length. Better to keep as INPUT_SEQUENCE_LENGTH+2 for start/end tokens
BATCH_SIZE = 192  # Default batch size which fits into 8GB of GPU memory
SHUFFLE_TRAINING_BATCHES = True  # Shuffle training batches in the dataset each epoch
EPOCHS_NUM = 100  # Total epochs num
GRAD_CLIP = 5.0  # Gradient clipping passed into theano.gradient.grad_clip()
LEARNING_RATE = 1.0  # Learning rate for the chosen optimizer (currently using Adadelta, see model.py)

# model params
NN_MODEL_PREFIX = 'cakechat'  # Specify prefix to be prepended to model's name

# predictions params
MAX_PREDICTIONS_LENGTH = 40  # Max. number of tokens which can be generated on the prediction step
PREDICTION_MODES = create_namedtuple_instance(
    'PREDICTION_MODES',
    beamsearch='beamsearch',
    beamsearch_reranking='beamsearch_reranking',
    sampling='sampling',
    sampling_reranking='sampling_reranking')
PREDICTION_MODE_FOR_TESTS = PREDICTION_MODES.sampling  # Default prediction mode used in metrics computation
PREDICTION_DISTINCTNESS_NUM_TOKENS = 50000  # Number of tokens which should be generated to compute distinctness metric

# Prediction probabilities modifiers
REPETITION_PENALIZE_COEFFICIENT = 10.0  # Divide the probabilities of the tokens already have been used during decoding
NON_PENALIZABLE_TOKENS = ['a', 'an', 'the', '*', '.', ',', '?', '!', '\'', '"', '^', '`']  # Exclude these tokens from
# repetition penalization modifier

# Options for sampling and sampling-reranking options
DEFAULT_TEMPERATURE = 0.5  # Default softmax temperature used for sampling

# Options for beamsearch and sampling-reranking:
BEAM_SIZE = 20  # Size of the beam (beamsearch only)
SAMPLES_NUM_FOR_RERANKING = 20  # Number of samples used in reranking (sampling-reranking only)
MMI_REVERSE_MODEL_SCORE_WEIGHT = 1.0  # Weight for MMI reranking reverse-model score, see the paper:
# 0.0 - scoring is performing using completely the default model, 1.0 - using completely the reverse model

# Logging params
LOG_CANDIDATES_NUM = 10  # Number of candidates to be printed to output during the logging
SCREEN_LOG_NUM_TEST_LINES = 10  # Number of first test lines to use when logging outputs on screen
SCREEN_LOG_FREQUENCY_PER_BATCHES = 500  # How many batches to train until next logging of output on screen
LOG_TO_TB_FREQUENCY_PER_BATCHES = 500  # How many batches to train until next metrics computed for TensorBoard
LOG_TO_FILE_FREQUENCY_PER_BATCHES = 2500  # How many batches to train until next logging of all the output into file
SAVE_MODEL_FREQUENCY_PER_BATCHES = 2500  # How many batches to train until next logging of all the output into file
AVG_LOSS_DECAY = 0.99  # Decay for the averaging the loss

# Use reduced sizes for input/output sequences, hidden layers and datasets sizes for the 'Developer Mode'
if is_dev_env():
    INPUT_SEQUENCE_LENGTH = 7
    OUTPUT_SEQUENCE_LENGTH = 9
    MAX_PREDICTIONS_LENGTH = 5
    BATCH_SIZE = 128
    HIDDEN_LAYER_DIMENSION = 7
    SCREEN_LOG_FREQUENCY_PER_BATCHES = 2
    LOG_TO_TB_FREQUENCY_PER_BATCHES = 3
    LOG_TO_FILE_FREQUENCY_PER_BATCHES = 4
    SAVE_MODEL_FREQUENCY_PER_BATCHES = 4
    WORD_EMBEDDING_DIMENSION = 15
    SAMPLES_NUM_FOR_RERANKING = BEAM_SIZE = 5
    LOG_CANDIDATES_NUM = 3
    USE_PRETRAINED_W2V_EMBEDDINGS_LAYER = False
    VAL_SUBSET_SIZE = 100
    MAX_VAL_LINES_NUM = 100
    TRAIN_SUBSET_SIZE = 10000
