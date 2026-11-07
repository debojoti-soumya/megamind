"""Universal Constructor "UCC Construction-1" indexer backend (hackathon sponsor).

STUB: access details are TBD (the coordinator will relay). When they arrive, implement
`retrieve` against their API behind the same `Retriever` interface and select it with
`RETRIEVER_BACKEND=ucc`. Until then this raises so misconfiguration is loud.
"""

from __future__ import annotations

from datetime import datetime

from megamind.config import Settings
from megamind.retrieval.base import RetrievalResult, RetrieveMode, Retriever


class UCCRetriever(Retriever):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult:
        raise NotImplementedError("UCCRetriever: waiting for UCC Construction-1 access details")
