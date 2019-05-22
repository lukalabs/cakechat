import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import set_keras_tf_session

gpu_memory_fraction = os.environ.get('GPU_MEMORY_FRACTION', 0.1)
set_keras_tf_session(gpu_memory_fraction)

from cakechat.api.v1.server import app

if __name__ == '__main__':
    # runs development server
    app.run(host='0.0.0.0', port=8080)
