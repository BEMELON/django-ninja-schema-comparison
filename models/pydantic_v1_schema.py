from typing import List
from pydantic.v1 import BaseModel as BaseModelV1
from .base import BaseMemberPrototype

class MemberBaseV1Schema(BaseModelV1, BaseMemberPrototype):
    id: int
    nickname: str
    
    class Config:
        orm_mode = True

class MemberListBaseV1Schema(BaseModelV1):
    member_list: List[MemberBaseV1Schema]
    
    class Config:
        orm_mode = True
