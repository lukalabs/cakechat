from operator import itemgetter

from cakechat.utils.tee_file import file_buffered_tee
from cakechat.utils.text_processing.config import DIALOG_CONDITION_FIELD, DIALOG_TEXT_FIELD
from cakechat.utils.text_processing.corpus_iterator import JsonTextLinesIterator


def get_flatten_dialogs(dialogs):
    for dialog in dialogs:
        for dialog_line in dialog:
            yield dialog_line


def get_alternated_dialogs_lines(dialogs):
    for dialog in dialogs:
        for first_dialog_line, second_dialog_line in zip(dialog, dialog[1:]):
            yield first_dialog_line
            yield second_dialog_line


def get_dialog_lines_and_conditions(dialog_lines, text_field_name, condition_field_name):
    """
    Splits one dialog_lines generator into two generators - one for conditions and one for dialog lines
    """
    conditions_iter, dialog_lines_iter = file_buffered_tee(
        map(lambda line: [line[condition_field_name], line[text_field_name]], dialog_lines))
    conditions_iter = map(itemgetter(0), conditions_iter)
    dialog_lines_iter = map(itemgetter(1), dialog_lines_iter)
    return dialog_lines_iter, conditions_iter


def load_processed_dialogs_from_json(lines, text_field_name, condition_field_name):
    for line_json in JsonTextLinesIterator(lines):
        yield [{
            text_field_name: entry[DIALOG_TEXT_FIELD],
            condition_field_name: entry[DIALOG_CONDITION_FIELD]
        } for entry in line_json]
