"""Loads Upstox credentials from the environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class UpstoxConfig:
    api_key: str | None
    api_secret: str | None
    redirect_uri: str | None
    access_token: str | None

    @property
    def auth_header(self) -> dict[str, str]:
        if not self.access_token:
            return {}
        return {"Authorization": f"Bearer {self.access_token}"}


def load_config() -> UpstoxConfig:
    return UpstoxConfig(
        api_key=os.getenv("UPSTOX_API_KEY"),
        api_secret=os.getenv("UPSTOX_API_SECRET"),
        redirect_uri=os.getenv("UPSTOX_REDIRECT_URI"),
        access_token=os.getenv("UPSTOX_ACCESS_TOKEN"),
    )
