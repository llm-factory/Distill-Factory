import json
from enum import Enum, unique
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel
from typing_extensions import Literal
from typing import List, Union


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class BaseMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]

    def __str__(self) -> str:
        if isinstance(self.content, str):
            content = self.content
        elif isinstance(self.content, list):
            content = json.dumps(self.content, ensure_ascii=False, indent=2)
        return f"role:{self.role} \ncontent:{content}"


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class UserMessage(BaseMessage):
    def __init__(self, content: Union[str, List[Dict[str, str]]]):
        super().__init__(role="user", content=content)


class AssistantMessage(BaseMessage):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: Optional[str] = None
    tool_calls: Optional[Iterable[Dict[str, Any]]] = None


class ToolMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(role="tool", content=content)


class FunctionAvailable(BaseModel):
    function: Dict[str, Any]
    type: Literal["function"] = "function"


def buildMessages(*messages: Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]) -> List[Dict[str, str]]:
    newMessages = []
    for message in messages:
        msg = {"role": message.role, "content": message.content}
        newMessages.append(msg)
    return newMessages
