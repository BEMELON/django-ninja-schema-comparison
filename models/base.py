from typing import List
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from ninja import Schema
from pydantic.utils import GetterDict
from django.db.models import Manager, QuerySet
from django.db.models.fields.files import FieldFile
from django.template import Variable, VariableDoesNotExist

# DjangoGetter from the original file
class DjangoGetter(GetterDict):
    __slots__ = ("_obj",)

    def __init__(self, obj: any):
        self._obj = obj

    def __getitem__(self, key: str) -> any:
        try:
            item = getattr(self._obj, key)
        except AttributeError:
            try:
                # Process attribute path separated by dots (e.g. "user.profile.name")
                item = Variable(key).resolve(self._obj)
            except VariableDoesNotExist as e:
                raise KeyError(key) from e
        return self._convert_result(item)

    def get(self, key: any, default: any = None) -> any:
        try:
            return self[key]
        except KeyError:
            return default

    def _convert_result(self, result: any) -> any:
        if isinstance(result, Manager):
            return list(result.all())

        elif isinstance(result, getattr(QuerySet, "__origin__", QuerySet)):
            return list(result)

        if callable(result) and not isinstance(result, type):
            return result()

        elif isinstance(result, FieldFile):
            if not result:
                return None
            return result.url

        return result

# Define SimpleDjangoSchema if not available
class SimpleDjangoSchema(BaseModel):
    """Simplified Django Schema class with resolver logic removed.
    Use for faster performance."""
    class Config:
        from_attributes = True
        getter_dict = DjangoGetter

# Base class for all MemberSchema classes (prototype)
class BaseMemberPrototype:
    id: int
    nickname: str
