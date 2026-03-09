from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UserEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    visitor_id: int
    event_type: str

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in {"view", "addtocart", "transaction"}:
            raise ValueError("event_type must be one of view, addtocart, transaction")
        return v

    item_id: int
    transaction_id: Optional[int] = None

    def to_json(self) -> str:
        return self.model_dump_json()
