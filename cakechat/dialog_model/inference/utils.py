import numpy as np

from cakechat.config import BATCH_SIZE, DEFAULT_CONDITION, INTX
from cakechat.dialog_model.model_utils import get_training_batch


def _predict_batch_by_batch(predict_fn, batched_inputs, non_batched_inputs=None, batch_size=BATCH_SIZE, num_outputs=1):
    """
    Splits prediction for big dataset in order to save GPU memory.
    Equivalent to predict_fn(*batched_inputs + non_batched_inputs).

    predict_fn: compiled keras predict function.
    batched_inputs:
        Inputs that we split into batches. On each iteration, we only pass one batch of this data into predict_fn.
    non_batched_inputs:
        Inputs that we do not split into batches. These inputs are the same for each call of predict_fn
    batch_size: int
        Size of each batch that we split our batched_inputs into
    num_ouputs: int, default=1
        Number of items returned on each call of predict_fn.
    """
    if non_batched_inputs is None:
        non_batched_inputs = []

    results = [[] for _ in range(num_outputs)]

    for inputs_batch in get_training_batch(batched_inputs, batch_size):
        args = list(inputs_batch) + non_batched_inputs
        cur_result = predict_fn(*args)
        if num_outputs > 1:
            for i in range(num_outputs):
                results[i].append(cur_result[i])
        else:
            results[0].append(cur_result)

    if num_outputs > 1:
        return tuple(np.concatenate(results[i]) for i in range(num_outputs))
    else:
        return np.concatenate(results[0])


def _handle_condition_ids(condition_ids, condition_to_index, num_responses):
    """
    Returns condition_ids preprocessed to match the shape of responses batch.
    Specifically:
        If condition_ids is None it is replaced with default condition index repeated num_responses times.
        If condition_ids is an one index, it is repeated num_responses times.
        If condition_ids is an array, assert that the shape is right.
    """
    if condition_ids is None:
        return np.array([condition_to_index[DEFAULT_CONDITION]] * num_responses, dtype=INTX)

    condition_ids = np.array(condition_ids, dtype=INTX)
    if len(condition_ids.shape) == 0:  # If condition_ids is one number
        return np.repeat(condition_ids[np.newaxis], num_responses, axis=0)
    elif condition_ids.shape != (num_responses, ):
        raise ValueError('Shape of condition_ids is {} and is not equal to (num_resonses, )={}'.format(
            condition_ids.shape, (num_responses, )))
    else:
        return condition_ids


def _predict_one_step(predict_fn,
                      condition_to_index,
                      thought_vectors,
                      prev_hidden_states,
                      prev_tokens_ids,
                      condition_ids=None,
                      temperature=1.0):
    condition_ids = _handle_condition_ids(condition_ids, condition_to_index, thought_vectors.shape[0])
    # We need newaxis to match the expected shape of an argument passed to predict_fn function
    prev_tokens_ids = prev_tokens_ids[:, np.newaxis]

    hidden_states, token_scores = \
        _predict_batch_by_batch(
            predict_fn,
            batched_inputs=[thought_vectors, prev_hidden_states, prev_tokens_ids, condition_ids],
            non_batched_inputs=[temperature],
            num_outputs=2)

    # token_scores is batch_size x num_tokens x vocab_size.
    # num_tokens is always 1, so we slice out the corresponding dimensionality.
    return hidden_states, token_scores[:, 0, :]


def get_sequence_score_by_thought_vector(nn_model, thought_vectors, y_ids, condition_ids=None):
    num_responses = thought_vectors.shape[0]
    condition_ids = _handle_condition_ids(condition_ids, nn_model.condition_to_index, num_responses)
    return _predict_batch_by_batch(
        nn_model.predict_sequence_score_by_thought_vector, batched_inputs=[thought_vectors, y_ids, condition_ids])


def get_sequence_score(nn_model, x_ids, y_ids, condition_ids=None):
    num_responses = x_ids.shape[0]
    condition_ids = _handle_condition_ids(condition_ids, nn_model.condition_to_index, num_responses)
    return _predict_batch_by_batch(nn_model.predict_sequence_score, batched_inputs=[x_ids, y_ids, condition_ids])


def get_sequence_log_probs(nn_model, x_ids, y_ids, condition_ids=None):
    num_responses = x_ids.shape[0]
    condition_ids = _handle_condition_ids(condition_ids, nn_model.condition_to_index, num_responses)
    return _predict_batch_by_batch(nn_model.predict_log_prob, batched_inputs=[x_ids, y_ids, condition_ids])


def get_thought_vectors(nn_model, x_ids):
    return _predict_batch_by_batch(nn_model.get_thought_vectors, batched_inputs=[x_ids])


def get_next_token_prob_one_step(nn_model, thoughts_batch, hidden_states_batch, prev_tokens_ids, condition_ids,
                                 **kwargs):
    return _predict_one_step(nn_model.predict_prob_one_step, nn_model.condition_to_index, thoughts_batch,
                             hidden_states_batch, prev_tokens_ids, condition_ids, **kwargs)


def get_next_token_log_prob_one_step(nn_model, thoughts_batch, hidden_states_batch, prev_tokens_ids, condition_ids,
                                     **kwargs):
    return _predict_one_step(nn_model.predict_log_prob_one_step, nn_model.condition_to_index, thoughts_batch,
                             hidden_states_batch, prev_tokens_ids, condition_ids, **kwargs)
