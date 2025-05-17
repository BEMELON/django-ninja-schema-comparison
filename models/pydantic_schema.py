from typing import List
from pydantic import BaseModel
from .base import BaseMemberPrototype

class MemberBaseSchema(BaseModel, BaseMemberPrototype):
    id: int
    nickname: str
    
    class Config:
        from_attributes = True

class MemberListBaseSchema(BaseModel):
    member_list: List[MemberBaseSchema]
    
    class Config:
        from_attributes = True
