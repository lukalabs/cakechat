import numpy as np
from sklearn.metrics import average_precision_score


def compute_average_precision(expected_answers, weighted_actual_answers, top):
    actual_responses, actual_weights = zip(*weighted_actual_answers.items())

    expected_labels = [int(response in expected_answers) for response in actual_responses][:top]
    actual_weights = actual_weights[:top]

    if any(expected_labels):
        score = average_precision_score(expected_labels, actual_weights)
    else:
        score = 0.0

    return score


def compute_recall_k(expected_answers, weighted_actual_answers, k):
    sorted_k_responses = sorted(
        weighted_actual_answers.keys(), key=lambda response: weighted_actual_answers[response], reverse=True)[:k]

    recall_k = len(set(sorted_k_responses) & set(expected_answers)) / len(expected_answers)
    return recall_k


def compute_retrieval_metric_mean(metric_func, questions_answers, questions_to_weighted_actual_answers, top_count):
    if top_count <= 0:
        raise ValueError('top_count should be a natural number')

    return np.mean([
        metric_func(answers, questions_to_weighted_actual_answers[question], top_count)
        for question, answers in questions_answers.items()
    ])
