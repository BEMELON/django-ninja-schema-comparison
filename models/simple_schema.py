from typing import List
from .base import SimpleDjangoSchema, BaseMemberPrototype

class MemberSimpleSchema(SimpleDjangoSchema, BaseMemberPrototype):
    id: int
    nickname: str
    
    class Config:
        orm_mode = True

class MemberListSimpleSchema(SimpleDjangoSchema):
    member_list: List[MemberSimpleSchema]
    
    class Config:
        orm_mode = True
