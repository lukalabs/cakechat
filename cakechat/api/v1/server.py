from flask import Flask, request, jsonify

from cakechat.api.response import get_response
from cakechat.api.utils import get_api_error_response, parse_dataset_param
from cakechat.config import EMOTIONS_TYPES, DEFAULT_CONDITION
from cakechat.utils.logger import get_logger
from cakechat.utils.profile import timer

_logger = get_logger(__name__)

app = Flask(__name__)


@app.route('/cakechat_api/v1/actions/get_response', methods=['POST'])
@timer
def get_model_response():
    params = request.get_json()
    _logger.info('request params: %s' % params)

    try:
        dialog_context = parse_dataset_param(params, param_name='context')
    except KeyError as e:
        return get_api_error_response('Malformed request, no "%s" param was found' % str(e), 400, _logger)
    except ValueError as e:
        return get_api_error_response('Malformed request: %s' % str(e), 400, _logger)

    emotion = params.get('emotion', DEFAULT_CONDITION)
    if emotion not in EMOTIONS_TYPES:
        return get_api_error_response('Malformed request, emotion param "%s" is not in emotion list %s' %
                                      (emotion, list(EMOTIONS_TYPES)), 400, _logger)

    response = get_response(dialog_context, emotion)

    if not response:
        _logger.error('No response for context: %s; emotion "%s"' % (dialog_context, emotion))
        return jsonify({}), 200

    _logger.info('Given response: "%s" for context: %s; emotion "%s"' % (response, dialog_context, emotion))

    return jsonify({'response': response}), 200
