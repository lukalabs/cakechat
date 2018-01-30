import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.api.v1.server import app

if __name__ == '__main__':
    # runs development server
    app.run(host='0.0.0.0', port=8080)
