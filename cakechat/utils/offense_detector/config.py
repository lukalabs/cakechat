import os
import pkg_resources

import cakechat.utils.offense_detector

OFFENSIVE_PHRASES_PATH = pkg_resources.resource_filename(cakechat.utils.offense_detector.__name__,
                                                         os.path.join('data', 'offensive_phrases.csv'))

