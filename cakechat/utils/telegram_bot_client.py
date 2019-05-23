from abc import ABCMeta, abstractmethod

import telepot
import telepot.loop

from cakechat.utils.logger import WithLogger


class AbstractTelegramChatSession(WithLogger, metaclass=ABCMeta):
    """
    Specific implementations of a chat session should overload default
    message handler `default_handle_message`, and possibly some of specific
    handler like `handle_photo_message` or `handle_text_message`.

    `_register_command` can be used to register a specific command to be
    recognized by the bot along with a corresponding handler.

    For example consider the following chat session that just echoes
    the input in a reverse order

    class ReversedChatSession(AbstractTelegramChatSession):
        def __init__(self, *args, **kwargs):
            super(ReversedChatSession, self).__init__(*args, **kwargs)
            self._register_command('start', self.greet)

        def greet(self, arg):
            self._send_text('!olleH')

        def handle_text_message(self, msg_text, msg):
            self._send_text(msg_text[::-1])

        def default_handle_message(self, msg):
            self._send_text('Sorry, I can only reverse text')

    In order to run a telegram bot with a given ChatSession, create
    an instance of TelegramBot and call the run method.

    TelegramBot(token).run(ReversedChatSession)
    """

    def __init__(self, bot, chat_id):
        super(AbstractTelegramChatSession, self).__init__()
        self._bot = bot
        self._chat_id = chat_id

        self._command_to_handler = {}
        # Default no-op /start command handler
        self._register_command(command='start', handler=lambda *args: None, description='Start or restart a session')
        self._register_command(
            command='help', handler=self._send_bot_help, description='Show bot info and a list of available commands')

    def _send_text(self, text, *args, **kwargs):
        self._bot.sendMessage(self._chat_id, text, *args, **kwargs)

    def _send_photo(self, photo, *args, **kwargs):
        self._bot.sendPhoto(self._chat_id, photo, *args, **kwargs)

    def handle_text_message(self, text, msg):
        self._logger.info('Received text message ({}), falling back to the default message handler'.format(text))
        return self.default_handle_message(msg)

    def handle_photo_message(self, photo_url, msg):
        self._logger.info('Received photo (url={}), falling back to the default message handler'.format(photo_url))
        return self.default_handle_message(msg)

    @abstractmethod
    def default_handle_message(self, msg):
        pass

    @staticmethod
    def _bot_info():
        return 'This is Replika CakeChat telegram bot.'

    def _send_bot_help(self, _):
        help_lines = [self._bot_info(), '', 'List of available commands:']
        for command, (_, description) in self._command_to_handler.items():
            help_lines.append('/{} - {}'.format(command, description))

        return self._send_text('\n'.join(help_lines))

    def handle_command(self, command, arg):
        if command not in self._command_to_handler:
            self._send_text('Unknown command {}'.format(command))
            return self._send_bot_help(arg)

        handler, _ = self._command_to_handler[command]
        return handler(arg)

    def _register_command(self, command, handler, description='???'):
        self._command_to_handler[command] = (handler, description)


class TelegramBot(WithLogger):
    """
    Interface for telegram bot API. Each bot maintains many
    ChatSessions that are handled by classes
    """

    def __init__(self, token):
        """

        :param token: a bot authorization token, can be obtained from the @BotFather bot.
        """
        super(TelegramBot, self).__init__()

        self._token = token
        self._bot = telepot.Bot(token)
        self._chat_id_to_session = {}

    @staticmethod
    def _parse_command(msg_text):
        """
        Parse message text as /<command> [argument]
        Further parsing of `argument` is left for specific
        implementations of `AbstractTelegramChatSession.handle_command`
        """
        if not msg_text.startswith('/'):
            raise ValueError('The command must start with /')

        command_and_arg = msg_text[1:].strip().split(' ', 1)
        command = command_and_arg[0]
        arg = command_and_arg[1] if len(command_and_arg) > 1 else ''
        return command, arg

    def _extract_photo_url(self, photo_sizes):
        # Telegram prepares several resized versions of the image,
        # we chose the biggest, assuming it's the original one
        photo_id = max(photo_sizes, key=lambda x: x['width'] * x['height'])['file_id']
        photo_path = self._bot.getFile(photo_id)['file_path']
        return 'https://api.telegram.org/file/bot{token}/{path}'.format(token=self._token, path=photo_path)

    def _init_chat_session(self, chat_id, session_class, **session_kwargs):
        session = session_class(self._bot, chat_id, **session_kwargs)
        action = 'Started new' if chat_id not in self._chat_id_to_session else 'Restarted existing'
        self._logger.info('{} chat session {}, currently opened: {}'.format(action, chat_id,
                                                                            len(self._chat_id_to_session)))
        return session

    def run(self, session_class, **session_kwargs):
        """
        :param session_class: subclass of AbstractTelegramChat
        """
        self._logger.info('Started a new chat bot {}'.format(self._bot.getMe()))

        def _handler(msg):
            content_type, chat_type, chat_id = telepot.glance(msg)

            if chat_id not in self._chat_id_to_session:
                self._chat_id_to_session[chat_id] = self._init_chat_session(chat_id, session_class, **session_kwargs)

            session = self._chat_id_to_session[chat_id]

            if content_type == 'text' and msg['text'].startswith('/'):
                command, arg = self._parse_command(msg['text'])
                if command == 'start':
                    session = self._chat_id_to_session[chat_id] = self._init_chat_session(
                        chat_id, session_class, **session_kwargs)

                return session.handle_command(command, arg)

            if content_type == 'text':
                return session.handle_text_message(msg['text'], msg)

            if content_type == 'photo':
                photo_url = self._extract_photo_url(msg['photo'])
                return session.handle_photo_message(photo_url, msg)

            return session.default_handle_message(msg)

        telepot.loop.MessageLoop(self._bot, _handler).run_forever()
