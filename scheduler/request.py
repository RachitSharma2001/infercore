import numpy as np
from concurrent.futures import Future

import queue 

DONE_TOKEN="<eos>"

class Request:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.future = Future()
        self.resp_tokens = queue.Queue()

