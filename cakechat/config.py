import os

from cakechat.utils.data_structures import create_namedtuple_instance
from cakechat.utils.env import is_dev_env

MODEL_NAME = 'cakechat_v2.0_keras_tf'

INTX = 'uint16'  # use unsigined 16-bits int representation for memory efficiency
RANDOM_SEED = 42  # Fix the random seed to a certain value to make everything reproducible

# AWS S3 params
S3_MODELS_BUCKET_NAME = 'cake-chat-data-v2'  # S3 bucket with all the data
S3_NN_MODEL_REMOTE_DIR = 'nn_models'  # S3 remote directory with models itself
S3_TOKENS_IDX_REMOTE_DIR = 'tokens_index'  # S3 remote directory with tokens index
S3_CONDITIONS_IDX_REMOTE_DIR = 'conditions_index'  # S3 remote directory with conditions index
S3_W2V_REMOTE_DIR = 'w2v_models'  # S3 remote directory with pre-trained w2v models

# train datasets
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PROCESSED_CORPUS_DIR = os.path.join(DATA_PATH, 'corpora_processed')
TOKEN_INDEX_DIR = os.path.join(DATA_PATH, 'tokens_index')  # Path to prepared tokens index directory
CONDITION_IDS_INDEX_DIR = os.path.join(DATA_PATH, 'conditions_index')  # Path to prepared conditions index directory

# train & val data params
BASE_CORPUS_NAME = 'processed_dialogs'  # Basic corpus name prefix
TRAIN_CORPUS_NAME = 'train_' + BASE_CORPUS_NAME  # Training dataset filename prefix
CONTEXT_SENSITIVE_VAL_CORPUS_NAME = 'val_' + BASE_CORPUS_NAME  # Validation dataset filename prefix for intermediate
CONTEXT_SENSITIVE_TEST_CORPUS_NAME = 'test_' + BASE_CORPUS_NAME  # Testing dataset for final metrics calculation
MAX_VAL_LINES_NUM = 10000  # Max lines number from validation set to be used for metrics calculation

# test datasets
TEST_DATA_DIR = os.path.join(DATA_PATH, 'quality')
CONTEXT_FREE_VAL_CORPUS_NAME = 'context_free_validation_set'  # Context-free validation set path
TEST_CORPUS_NAME = 'context_free_test_set'  # Context-free test set path
QUESTIONS_CORPUS_NAME = 'context_free_questions'  # Context-free questions only path

# directory to store model wights and calcualted metrics
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')  # Directory to store training results
TENSORBOARD_LOG_DIR = os.path.join(RESULTS_PATH, 'tensorboard')  # Path to tensorboard logs directory

# word embeddings params
USE_PRETRAINED_W2V_EMBEDDINGS_LAYER = True  # Whether to use word2vec to pre-train weights for the embedding layer
TRAIN_WORD_EMBEDDINGS_LAYER = True  # Allow fine-tuning of the word embedding layer during the model training
W2V_MODEL_DIR = os.path.join(DATA_PATH, 'w2v_models')  # Path to store & load trained word2vec models
WORD_EMBEDDING_DIMENSION = 128  # word2vec embedding dimension
W2V_WINDOW_SIZE = 10  # word2vec window size, used during the w2v pre-training
USE_SKIP_GRAM = True  # Use skip-gram word2vec mode. When False, CBOW is used
TOKEN_REPRESENTATION_SIZE = 256
MIN_WORD_FREQ = 1  # Minimum frequency of a word to be used in word2vec pre-calculation
VOCABULARY_MAX_SIZE = 50000  # Maximum vocabulary size in tokens
MAX_CONDITIONS_NUM = 5  # Maximum conditions num

# condition inputs. We use five major emotions to condition our model's predictions
EMOTIONS_TYPES = create_namedtuple_instance(
    'EMOTIONS_TYPES', neutral='neutral', anger='anger', joy='joy', fear='fear', sadness='sadness')
DEFAULT_CONDITION = EMOTIONS_TYPES.neutral  # Default condition to be used during the prediction (if not specified)
CONDITION_EMBEDDING_DIMENSION = 128  # Conditions embedding layer dimension to be trained.

# NN architecture params
HIDDEN_LAYER_DIMENSION = 768  # Dimension for the recurrent layer
DENSE_DROPOUT_RATIO = 0.2  # Use dropout with the given ratio before decoder's output
USE_CUDNN = bool(os.environ.get('CUDA_VISIBLE_DEVICES'))  # True by default for GPU-enable machines (provides ~25% inference
# speed up) and False on CPU-only machines since they does not support CuDNN

# training params
EPOCHS_NUM = 2  # Total epochs num
BATCH_SIZE = 196  # Number of samples to be used for gradient estimation on each train step. In case of using multiple
# GPUs for train, each worker will have this number of samples on each step.
SHUFFLE_TRAINING_BATCHES = True  # Shuffle training batches in the dataset each epoch

INPUT_SEQUENCE_LENGTH = 30  # Input sequence length for the model during the training;
INPUT_CONTEXT_SIZE = 3  # Maximum depth of the conversational history to be used in encoder (at least 1)
OUTPUT_SEQUENCE_LENGTH = 32  # Output sequence length. Better to keep as INPUT_SEQUENCE_LENGTH+2 for start/end tokens

GRAD_CLIP = 5.0  # Gradient clipping param passed to optimizer
LEARNING_RATE = 6.0  # Learning rate for Adadelta optimzer
LOG_RUN_METADATA = False  # Set 'True' to profile memory consumption and computation time on tensorboard
AUTOENCODER_MODE = False  # Set 'True' to switch seq2seq (x -> y) into autoencoder (x -> x). Used for debugging

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
BEAM_SIZE = 10  # Size of the beam (beamsearch only)
SAMPLES_NUM_FOR_RERANKING = 10  # Number of samples used in reranking (sampling-reranking only)
MMI_REVERSE_MODEL_SCORE_WEIGHT = 1.0  # Weight for MMI reranking reverse-model score, see the paper:
# 0.0 - scoring is performing using completely the default model, 1.0 - using completely the reverse model

# Logging params
LOG_CANDIDATES_NUM = 3  # Number of candidates to be printed to output during the logging
SCREEN_LOG_NUM_TEST_LINES = 10  # Number of first test lines to use when logging outputs on screen
EVAL_STATE_PER_BATCHES = 500  # How many batches to train until next metrics computed for TensorBoard

# Use reduced params values for development
if is_dev_env():
    # train & val data params
    MAX_VAL_LINES_NUM = 10

    # word embeddings params
    USE_PRETRAINED_W2V_EMBEDDINGS_LAYER = True
    TRAIN_WORD_EMBEDDINGS_LAYER = True
    WORD_EMBEDDING_DIMENSION = 64
    VOCABULARY_MAX_SIZE = 1000
    MAX_CONDITIONS_NUM = 5

    # condition inputs
    CONDITION_EMBEDDING_DIMENSION = 1

    # NN architecture params
    HIDDEN_LAYER_DIMENSION = 128
    DENSE_DROPOUT_RATIO = 0.2
    USE_CUDNN = False

    # training params
    INPUT_SEQUENCE_LENGTH = 3
    INPUT_CONTEXT_SIZE = 1
    OUTPUT_SEQUENCE_LENGTH = 5
    BATCH_SIZE = 4
    SHUFFLE_TRAINING_BATCHES = False
    EPOCHS_NUM = 4
    LEARNING_RATE = 1.0
    LOG_RUN_METADATA = False
    AUTOENCODER_MODE = False

    # predictions params
    MAX_PREDICTIONS_LENGTH = 4

    # options for beamsearch and sampling-reranking:
    SAMPLES_NUM_FOR_RERANKING = 5
    BEAM_SIZE = 5

    # logging params
    LOG_CANDIDATES_NUM = 3
    SCREEN_LOG_NUM_TEST_LINES = 4
    EVAL_STATE_PER_BATCHES = 5
