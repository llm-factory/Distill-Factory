import unittest

from ..model.chat_openai import ChatOpenAI


class TestChat(unittest.TestCase):
    def test_chat(self):
        message = "hello"
        response = chat_openai.chat([{"role": "user", "content": message}])
        print(response)
        self.assertIsInstance(response, str)
