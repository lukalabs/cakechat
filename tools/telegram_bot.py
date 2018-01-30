import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import deque

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.api.response import get_response
from cakechat.config import INPUT_CONTEXT_SIZE, DEFAULT_CONDITION
from cakechat.utils.telegram_bot_client import TelegramBot, AbstractTelegramChatSession


class CakeChatTelegramChatSession(AbstractTelegramChatSession):
    def __init__(self, *args, **kwargs):
        super(CakeChatTelegramChatSession, self).__init__(*args, **kwargs)
        self._context = deque(maxlen=INPUT_CONTEXT_SIZE)

    def handle_text_message(self, msg_text, msg):
        self._context.append(msg_text.strip())
        response = get_response(self._context, DEFAULT_CONDITION)
        self._context.append(response)
        self._send_text(response)

    def default_handle_message(self, msg):
        self._send_text('Sorry bruh, text only')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--token', help='Bot token')
    args = argparser.parse_args()

    TelegramBot(token=args.token).run(CakeChatTelegramChatSession)
