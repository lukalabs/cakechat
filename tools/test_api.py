from __future__ import print_function
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from cakechat.config import DEFAULT_CONDITION

_HOST_FQDN = '127.0.0.1'
_SERVER_PORT = '8080'


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--fqdn', action='store', default=_HOST_FQDN)
    argparser.add_argument('-p', '--port', action='store', default=_SERVER_PORT)
    argparser.add_argument('-c', '--context', action='append', help='set "-c your_dialog_context"', required=True)
    argparser.add_argument('-e', '--emotion', action='store', default=DEFAULT_CONDITION)

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    url = 'http://%s:%s/cakechat_api/v1/actions/get_response' % (args.fqdn, args.port)
    body = {'context': args.context, 'emotion': args.emotion}

    response = requests.post(url, json=body)
    print(response.json())
