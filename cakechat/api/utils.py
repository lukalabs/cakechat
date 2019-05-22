from flask import jsonify


def get_api_error_response(message, code, logger):
    logger.error(message)
    return jsonify({'message': message}), code


def _is_list_of_unicode_strings(data):
    return data and isinstance(data, (list, tuple)) and all(isinstance(s, str) for s in data)


def parse_dataset_param(params, param_name, required=True):
    if not required and params.get(param_name) is None:
        return None

    dataset = params[param_name]
    if not _is_list_of_unicode_strings(dataset):
        raise ValueError('`{}` should be non-empty list of unicode strings'.format(param_name))
    if not all(dataset):
        raise ValueError('`{}` should not contain empty strings'.format(param_name))

    return dataset
