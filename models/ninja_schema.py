from typing import List
from ninja import Schema
from .base import BaseMemberPrototype

class MemberSchema(Schema, BaseMemberPrototype):
    id: int
    nickname: str
    
    class Config:
        orm_mode = True

class MemberListSchema(Schema):
    member_list: List[MemberSchema]
    
    class Config:
        orm_mode = True
