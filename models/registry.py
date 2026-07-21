from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.types import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    name: str
    skills: list[str]
    options: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Loads model config and validates models exist in Ollama at startup."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._ollama_host: str = config.get("ollama_host", "http://localhost:11434")
        models_cfg: dict[str, Any] = config.get("models", config)

        coordinator_name: str | None = models_cfg.get("coordinator")
        default_name: str | None = models_cfg.get("default")

        if not coordinator_name:
            raise ValueError("models.coordinator is required in config")
        if not default_name:
            raise ValueError("models.default is required in config")

        self._coordinator = ModelSpec(name=coordinator_name)
        self._default = ModelSpec(name=default_name)
        self._embed_model: str = models_cfg.get("embed", "nomic-embed-text")

        raw_workers: list[dict[str, Any]] = models_cfg.get("workers", [])
        self._workers: list[ModelSpec] = [
            ModelSpec(
                name=w["name"],
                skills=w.get("skills", []),
                options=w.get("options", {}),
            )
            for w in raw_workers
        ]

    async def validate(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()

        available: set[str] = {m["name"] for m in data.get("models", [])}

        required: list[str] = [
            self._coordinator.name,
            self._default.name,
            self._embed_model,
            *(w.name for w in self._workers),
        ]

        missing = [name for name in required if name not in available]
        if missing:
            raise ValueError(
                f"The following models are not available in Ollama "
                f"(run `ollama pull <name>`): {missing}"
            )

        all_skills: list[str] = []
        for worker in self._workers:
            for skill in worker.skills:
                if skill in all_skills:
                    logger.warning(
                        "Duplicate skill tag %r found across workers", skill
                    )
                else:
                    all_skills.append(skill)

    @property
    def coordinator(self) -> ModelSpec:
        return self._coordinator

    @property
    def default(self) -> ModelSpec:
        return self._default

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @property
    def workers(self) -> list[ModelSpec]:
        return self._workers
