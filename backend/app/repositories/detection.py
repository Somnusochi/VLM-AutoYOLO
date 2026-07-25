"""Data-access layer for Detection / DetectionBox.

All DB queries live here — routes and services never touch Session directly.
"""

from __future__ import annotations

from collections.abc import Sequence
import uuid

from sqlalchemy.orm import Session

from ..models.detection import Detection, DetectionBox


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


class DetectionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── write ──────────────────────────────────────────

    def create(
        self,
        *,
        image_path: str,
        image_name: str,
        image_width: int,
        image_height: int,
        categories: list,  # JSONB stores Python list directly
        model_name: str = "LocateAnything-3B",
        commit: bool = False,
    ) -> Detection:
        det = Detection(
            image_path=image_path,
            image_name=image_name,
            image_width=image_width,
            image_height=image_height,
            categories=categories,
            model_name=model_name,
        )
        self.db.add(det)
        if commit:
            self.db.commit()
            self.db.refresh(det)
        else:
            self.db.flush()  # get id without committing
        return det

    def add_boxes(
        self, detection_id: str | uuid.UUID, boxes: list[dict], commit: bool = False
    ) -> list[DetectionBox]:
        """boxes: [{"x1","y1","x2","y2","class_name","confidence","mask_polygon"}, ...]"""
        detection_uuid = _as_uuid(detection_id)
        entities = [
            DetectionBox(
                detection_id=detection_uuid,
                class_name=b["class_name"],
                x1=b["x1"],
                y1=b["y1"],
                x2=b["x2"],
                y2=b["y2"],
                confidence=b.get("confidence"),
                mask_polygon=b.get("mask_polygon"),
            )
            for b in boxes
        ]
        self.db.add_all(entities)
        if commit:
            self.db.commit()
        else:
            self.db.flush()
        return entities

    def replace_boxes(
        self, detection_id: str | uuid.UUID, boxes: list[dict], commit: bool = False
    ) -> list[DetectionBox]:
        """Delete all existing boxes for a detection and insert new ones."""
        detection_uuid = _as_uuid(detection_id)
        self.db.query(DetectionBox).filter(
            DetectionBox.detection_id == detection_uuid,
        ).delete()
        # Note: If commit=True, the flush on delete is skipped, and the subsequent
        # add_boxes(commit=True) will commit both the delete and the inserts
        # in a single transaction.
        if not commit:
            self.db.flush()
        return self.add_boxes(detection_uuid, boxes, commit=commit)

    def update_detection(self, detection: Detection, commit: bool = False) -> None:
        if commit:
            self.db.commit()
            self.db.refresh(detection)
        else:
            self.db.flush()

    def update_box(self, box: DetectionBox, commit: bool = False) -> None:
        if commit:
            self.db.commit()
            self.db.refresh(box)
        else:
            self.db.flush()

    # ── read ───────────────────────────────────────────

    def get_by_id(self, detection_id: str | uuid.UUID) -> Detection | None:
        return self.db.query(Detection).filter(Detection.id == _as_uuid(detection_id)).first()

    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[Detection], int]:
        q = self.db.query(Detection)
        total = q.count()
        items = (
            q.order_by(Detection.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total

    # ── delete ─────────────────────────────────────────

    def delete(self, detection: Detection, commit: bool = False) -> None:
        self.db.delete(detection)
        if commit:
            self.db.commit()
        else:
            self.db.flush()

    def get_box(self, detection_id: str, box_id: str) -> DetectionBox | None:
        return (
            self.db.query(DetectionBox)
            .filter(
                DetectionBox.id == _as_uuid(box_id),
                DetectionBox.detection_id == _as_uuid(detection_id),
            )
            .first()
        )

    def delete_box(self, box: DetectionBox, commit: bool = False) -> None:
        self.db.delete(box)
        if commit:
            self.db.commit()
        else:
            self.db.flush()
