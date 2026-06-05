# SPDX-License-Identifier: MIT
"""GLOSSATE ASCII art and decorative elements."""

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .theme import Colors

# GLOSSATE in ANSI Shadow figlet style
GLOSSATE_LOGO = """
 ██████╗ ██╗      ██████╗ ███████╗███████╗ █████╗ ████████╗███████╗
██╔════╝ ██║     ██╔═══██╗██╔════╝██╔════╝██╔══██╗╚══██╔══╝██╔════╝
██║  ███╗██║     ██║   ██║███████╗███████╗███████║   ██║   █████╗
██║   ██║██║     ██║   ██║╚════██║╚════██║██╔══██║   ██║   ██╔══╝
╚██████╔╝███████╗╚██████╔╝███████║███████║██║  ██║   ██║   ███████╗
 ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""


GLOSSATE_LOGO_SMALL = (
    ",---.|    ,---.,---.,---.,---.|--- ,---.\n"
    "|   ||    |   |`---.`---.,---||    |---'\n"
    "`---'`---'`---'`---'`---'`---^`---'`---'"
)


def get_styled_logo() -> Text:
    """Return the GLOSSATE logo with a white-to-gray fade."""
    text = Text()
    lines = GLOSSATE_LOGO.strip("\n").split("\n")

    gradient = [
        Colors.PRIMARY,
        Colors.PRIMARY,
        Colors.PRIMARY_LIGHT,
        Colors.PRIMARY_LIGHT,
        Colors.PRIMARY_DARK,
        Colors.PRIMARY_DARK,
    ]

    for i, line in enumerate(lines):
        color = gradient[min(i, len(gradient) - 1)]
        text.append(line + "\n", style=color)

    return text


def get_header_panel(console_width: int = 80) -> Panel:
    """Generate the complete header with logo and subtitle, centered."""
    logo = get_styled_logo()

    subtitle = Text(justify="center")
    subtitle.append("GLOSSATE - Locally transcribe and translate", style=Colors.TEXT_MUTED)

    content = Group(Align.center(logo), Align.center(subtitle))

    return Panel(
        content,
        border_style=Colors.PRIMARY,
        padding=(1, 2),
    )
