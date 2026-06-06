from typing import Literal
from pydantic import BaseModel

Label = Literal["toxic", "not_toxic"]


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: Label


class ClassifyBatchRequest(BaseModel):
    texts: list[str]


class ClassifyBatchResponse(BaseModel):
    results: list[ClassifyResponse]
