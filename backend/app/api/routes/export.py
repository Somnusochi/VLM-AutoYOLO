from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from starlette.responses import Response

from ...core.database import get_db
from ...schemas.detection import ExportAllIn, ExportBatchIn
from ...services.export import FORMAT_LABELS, export_all, export_batch, export_single

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["export"])


@router.get("/detections/{detection_id}/export")
def download_single_yolo(
    detection_id: str,
    db: Session = Depends(get_db),
) -> PlainTextResponse:
    try:
        content, image_name = export_single(db, detection_id)
    except Exception as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    base = Path(image_name).stem
    return PlainTextResponse(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{base}.txt"'},
    )


@router.post("/detections/export-batch")
def download_batch_yolo(
    body: ExportBatchIn,
    db: Session = Depends(get_db),
) -> Response:
    fmt = body.format or "yolo"
    data = export_batch(db, body.detection_ids, format=fmt)
    label = FORMAT_LABELS.get(fmt, fmt)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{label}_dataset.zip"'},
    )


@router.post("/detections/export-all")
def download_all_detections(
    body: ExportAllIn,
    db: Session = Depends(get_db),
) -> Response:
    """Download every saved detection as one dataset archive."""
    try:
        data = export_all(db, format=body.format)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    label = FORMAT_LABELS.get(body.format, body.format)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{label}_all_dataset.zip"'},
    )
