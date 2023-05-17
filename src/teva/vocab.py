import os

import seqio
from dotenv import load_dotenv

load_dotenv()


DEFAULT_SPM_PATH = os.getenv("DEFAULT_SPM_PATH")

DEFAULT_VOCAB = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)
