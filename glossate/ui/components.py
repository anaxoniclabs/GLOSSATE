# SPDX-License-Identifier: MIT
"""Reusable UI components for GLOSSATE CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .theme import GLOSSATE_THEME, Colors


def create_console() -> Console:
    """Create a themed Rich console."""
    return Console(theme=GLOSSATE_THEME, highlight=False)


def info_panel(title: str, content: dict[str, str], style: str = Colors.PRIMARY) -> Panel:
    """Create a key-value info panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    for label, value in content.items():
        table.add_row(label, value)

    return Panel(
        table,
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2),
    )
