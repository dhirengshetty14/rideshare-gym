"""RideshareTask — base class for all 12 tasks."""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.task import AbstractTask
from rideshare_gym.rideshare_sandbox import RideshareSandbox


class RideshareTask(AbstractTask):
    """Adds RideshareSandbox typing on top of AbstractTask."""

    def setup(self, sandbox: RideshareSandbox) -> Any:  # type: ignore[override]
        raise NotImplementedError
