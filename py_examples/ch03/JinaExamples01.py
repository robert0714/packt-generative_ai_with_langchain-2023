from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage
from config import set_environment

set_environment()

chat = JinaChat(temperature=0.)
messages = [
    HumanMessage(
        content="Translate this sentence from English to French: I love generative AI!"
    )
]
chat(messages)