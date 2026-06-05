# SPDX-License-Identifier: MIT
"""Live terminal progress display for GLOSSATE's three-stage pipeline."""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .components import create_console
from .theme import BoxChars, Colors, Stages

_TOTAL_UNITS = 1000

_WEIGHTS_WITH_TRANSLATE    = {"extract": 0.03, "transcribe": 0.47, "translate": 0.50}
_WEIGHTS_WITHOUT_TRANSLATE = {"extract": 0.05, "transcribe": 0.95, "translate": 0.00}


class StageState(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETE = auto()
    SKIPPED = auto()
    ERROR = auto()


class _StageInfo:
    def __init__(self, stage_id: str, label: str, icon: str, past_label: str = "") -> None:
        self.stage_id = stage_id
        self.label = label
        self.past_label = past_label or label
        self.icon = icon
        self.state: StageState = StageState.PENDING
        self.error_message: str = ""


class GlossateProgress:
    """Single continuous progress bar across all pipeline stages."""

    def __init__(self, has_translate: bool = True) -> None:
        self._console = create_console()

        self._stages: dict[str, _StageInfo] = {
            sid: _StageInfo(sid, label, icon, past_label)
            for sid, label, icon, past_label in Stages.ALL
        }

        self._has_translate = has_translate
        weights = _WEIGHTS_WITH_TRANSLATE if has_translate else _WEIGHTS_WITHOUT_TRANSLATE
        offset = 0
        self._stage_offset: dict[str, int] = {}
        self._stage_units: dict[str, int] = {}
        for sid, w in weights.items():
            self._stage_offset[sid] = offset
            self._stage_units[sid] = round(w * _TOTAL_UNITS)
            offset += self._stage_units[sid]

        self._active_stage: Optional[str] = None
        self._active_local_total: int = 1

        self._file_title: str = ""
        self._file_duration: str = ""
        self._detail: str = ""

        self._progress = Progress(
            SpinnerColumn("bouncingBall"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task("", total=_TOTAL_UNITS, completed=0)
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=20,
        )
        self._started = False

    def _bullet(self, state: StageState) -> str:
        return {
            StageState.PENDING:  BoxChars.BULLET_PENDING,
            StageState.ACTIVE:   BoxChars.BULLET_ACTIVE,
            StageState.COMPLETE: BoxChars.BULLET_COMPLETE,
            StageState.SKIPPED:  BoxChars.BULLET_COMPLETE,
            StageState.ERROR:    BoxChars.BULLET_ERROR,
        }[state]

    def _bullet_style(self, state: StageState) -> str:
        return {
            StageState.PENDING:  Colors.TEXT_MUTED,
            StageState.ACTIVE:   Colors.PRIMARY,
            StageState.COMPLETE: Colors.SUCCESS,
            StageState.SKIPPED:  Colors.TEXT_MUTED,
            StageState.ERROR:    Colors.ERROR,
        }[state]

    def _render(self) -> Panel:
        lines: list[Text] = []

        if self._file_title:
            header = Text()
            header.append(self._file_title, style=f"bold {Colors.TEXT_PRIMARY}")
            if self._file_duration:
                header.append(f"  {self._file_duration}", style=Colors.TEXT_MUTED)
            lines.append(header)
            lines.append(Text(""))

        for stage in self._stages.values():
            if stage.stage_id == "translate" and not self._has_translate:
                continue
            bullet = self._bullet(stage.state)
            style = self._bullet_style(stage.state)
            line = Text()
            line.append(f"{bullet} ", style=style)
            if stage.icon:
                line.append(f"{stage.icon} ", style=style)
            done = stage.state in (StageState.COMPLETE, StageState.SKIPPED)
            line.append(stage.past_label if done else stage.label, style=style)
            if stage.state == StageState.ERROR and stage.error_message:
                line.append(f"  {stage.error_message}", style=Colors.ERROR)
            lines.append(line)

        content_parts: list[RenderableType] = [Text("\n").join(lines)]
        content_parts.append(Text(""))
        content_parts.append(self._progress)

        return Panel(
            Group(*content_parts),
            title="[bold]GLOSSATE[/bold]",
            border_style=Colors.PRIMARY,
            padding=(1, 2),
        )

    def _refresh(self) -> None:
        if self._started:
            self._live.update(self._render())

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._live.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._live.stop()

    def set_file_info(self, title: str, duration: str) -> None:
        self._file_title = title
        self._file_duration = duration
        self._refresh()

    def begin_stage(self, stage_id: str, total: int = 1) -> None:
        info = self._stages[stage_id]
        info.state = StageState.ACTIVE
        self._active_stage = stage_id
        self._active_local_total = max(total, 1)
        self._progress.update(
            self._task_id,
            completed=float(self._stage_offset[stage_id]),
            description=info.label,
        )
        self._refresh()

    def update_progress(self, completed: int, total: Optional[int] = None) -> None:
        if self._active_stage is None:
            return
        if total is not None:
            self._active_local_total = max(total, 1)
        local_frac = min(completed / self._active_local_total, 1.0)
        global_done = self._stage_offset[self._active_stage] + local_frac * self._stage_units[self._active_stage]

        stage_label = self._stages[self._active_stage].label
        if self._active_stage == "translate":
            desc = f"{stage_label}  {completed}/{self._active_local_total} cues"
        elif self._active_stage == "transcribe":
            m, s = divmod(int(completed), 60)
            tm, ts = divmod(int(self._active_local_total), 60)
            desc = f"{stage_label}  {m}:{s:02d} / {tm}:{ts:02d}"
        else:
            desc = stage_label

        self._progress.update(self._task_id, completed=global_done, description=desc)
        self._refresh()

    def complete_stage(self, stage_id: str) -> None:
        self._stages[stage_id].state = StageState.COMPLETE
        if self._active_stage == stage_id:
            self._active_stage = None
        end = float(self._stage_offset[stage_id] + self._stage_units[stage_id])
        self._progress.update(self._task_id, completed=end)
        self._refresh()

    def skip_stage(self, stage_id: str) -> None:
        self._stages[stage_id].state = StageState.SKIPPED
        end = float(self._stage_offset[stage_id] + self._stage_units[stage_id])
        self._progress.update(self._task_id, completed=end)
        self._refresh()

    def error_stage(self, stage_id: str, message: str = "") -> None:
        info = self._stages[stage_id]
        info.state = StageState.ERROR
        info.error_message = message
        if self._active_stage == stage_id:
            self._active_stage = None
        self._refresh()

    def get_state(self, stage_id: str) -> StageState:
        state = self._stages[stage_id].state
        return StageState.COMPLETE if state == StageState.SKIPPED else state
