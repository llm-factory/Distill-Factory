import os
import unittest

from model.config import Config
from api.api import API
from common.message import buildMessages, UserMessage



class TestChat(unittest.TestCase):
    def test_chat(self):
        message = "hello"
        api = API(Config("../src/model/config.yml"))
        response = api.chat([UserMessage(message)])
        print(response)
        self.assertIsInstance(response, str)
