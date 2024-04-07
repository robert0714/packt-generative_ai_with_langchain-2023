from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

from config import set_environment

set_environment()


chat = JinaChat(temperature=0.)
chat(
    [
        SystemMessage(
            content="You help a user find a nutritious and tasty food to eat in one word."
        ),
        HumanMessage(
            content="I like pasta with cheese, but I need to eat more vegetables, what should I eat?"
        )
    ]
)