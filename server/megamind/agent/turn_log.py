"""Every chat turn's question and answer with wall-clock times (`GET /v1/turns`), so a recording of the headset
view can be captioned with what was really asked and answered (tools/make_captions.py)."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()


class Turn(BaseModel):
    started_at: float  # request received (epoch s); a voice question was spoken just before this
    ended_at: float  # answer ready (text + audio)
    question: str  # typed text or the speech-to-text transcript
    answer: str
    voice: bool = False
    session_id: str = "default"
    device_id: str | None = None


class TurnsResponse(BaseModel):
    turns: list[Turn]  # oldest first


class TurnLog:
    """The newest `keep` turns in memory, plus one JSON line per turn appended to `path` (survives restarts)."""

    def __init__(self, path: Path | None, keep: int = 1000) -> None:
        self.path = path
        self._turns: deque[Turn] = deque(maxlen=keep)
        if path is not None and path.exists():
            for line in path.read_text().splitlines()[-keep:]:
                try:
                    self._turns.append(Turn.model_validate_json(line))
                except ValueError:
                    continue

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(turn.model_dump_json() + "\n")
        except OSError:
            log.warning("could not append to the turn log %s", self.path, exc_info=True)

    def query(self, since: float | None = None, until: float | None = None, limit: int = 200) -> list[Turn]:
        out = [
            t
            for t in self._turns
            if (since is None or t.ended_at >= since) and (until is None or t.started_at <= until)
        ]
        return out[-limit:]


@router.get("/turns", response_model=TurnsResponse)
async def list_turns(
    request: Request,
    since: float | None = Query(None, description="epoch s: turns that ended at or after this"),
    until: float | None = Query(None, description="epoch s: turns that started at or before this"),
    device_id: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
):
    turn_log: TurnLog | None = getattr(request.app.state, "turn_log", None)
    if turn_log is None:
        return TurnsResponse(turns=[])
    turns = turn_log.query(since, until, limit=1000)
    if device_id is not None:
        turns = [t for t in turns if t.device_id == device_id]
    return TurnsResponse(turns=turns[-limit:])


def record(app_state, started_at: float, ended_at: float, inp, answer: str) -> None:
    turn_log: TurnLog | None = getattr(app_state, "turn_log", None)
    if turn_log is None:
        return
    turn_log.add(
        Turn(
            started_at=started_at,
            ended_at=ended_at,
            question=inp.text,
            answer=answer,
            voice=inp.voice,
            session_id=inp.session_id,
            device_id=inp.device_id,
        )
    )
