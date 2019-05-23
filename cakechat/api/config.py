from cakechat.config import PREDICTION_MODES

# Prediction mode used in API
PREDICTION_MODE = PREDICTION_MODES.sampling_reranking

# In case of PREDICTION_MODES.{beamsearch, beamsearch-reranking, sampling-reranking} choose random non-offensive
# response out of K best candidates proposed by the algorithm.
NUM_BEST_CANDIDATES_TO_PICK_FROM = 3

# In case of PREDICTION_MODES.sampling generate samples one-by-one until a non-offensive sample occurs. This parameter
# defines max number of samples will be generated until succeed.
SAMPLING_ATTEMPTS_NUM = 10

# Default response text in case we weren't able to produce appropriate response.
DEFAULT_RESPONSE = '🙊'
