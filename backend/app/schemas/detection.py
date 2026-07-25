from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from ..models.detection import DetectionStatus, FilterMode
from .common import BaseSchema, _coerce_uuid


class DetectionParams(BaseSchema):
    use_sam2: bool = False
    use_sam3: bool = False
    sam2_score_threshold: float = 0.0
    sam3_text: str = ""
    use_sam3_seg: bool = True
    sam3_threshold: float = 0.5
    sam3_mask_threshold: float = 0.5


class DetectionBoxOut(BaseSchema):
    id: str
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None = None
    mask_polygon: list | None = None

    model_config = {"from_attributes": True}

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v: Any) -> str:
        return _coerce_uuid(v)


class DetectionBoxItem(BaseSchema):
    """Lightweight box without mask_polygon for list views."""

    id: str
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None = None

    model_config = {"from_attributes": True}

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v: Any) -> str:
        return _coerce_uuid(v)


class DetectionListItem(BaseSchema):
    """Lightweight detection for list endpoint — boxes without mask polygons."""

    id: str
    image_name: str
    categories: list[str] = []
    model_name: str
    model_type: str | None = None
    image_width: int
    image_height: int
    elapsed_ms: int | None = None
    filter_mode: FilterMode | None = None
    filter_nms_iou: float | None = None
    status: DetectionStatus
    created_at: datetime
    boxes: list[DetectionBoxItem] = []

    model_config = {"from_attributes": True}

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v: Any) -> str:
        return _coerce_uuid(v)


class DetectionOut(DetectionListItem):
    """Full detection with mask polygons — for detail endpoint."""

    boxes: list[DetectionBoxOut] = []


class ExportBatchIn(BaseSchema):
    detection_ids: list[str] = Field(..., min_length=1)
    format: str = "yolo"


class ExportAllIn(BaseSchema):
    format: str = "yolo-seg"
