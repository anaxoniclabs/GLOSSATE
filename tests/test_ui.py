# SPDX-License-Identifier: MIT
import io

from rich.console import Console
from rich.panel import Panel

from glossate.ui.ascii_art import GLOSSATE_LOGO_SMALL, get_header_panel, get_styled_logo
from glossate.ui.components import info_panel
from glossate.ui.theme import GLOSSATE_THEME, Colors, Stages


def _render(renderable: object) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=True, no_color=True)
    console.print(renderable)
    return buf.getvalue()


class TestColors:
    def test_primary_is_white(self) -> None:
        assert Colors.PRIMARY == "#FFFFFF"


class TestStages:
    def test_all_contains_three_stages(self) -> None:
        assert len(Stages.ALL) == 3

    def test_stage_ids(self) -> None:
        ids = [s[0] for s in Stages.ALL]
        assert ids == ["extract", "transcribe", "translate"]


class TestTheme:
    def test_glossate_primary_style_present(self) -> None:
        assert "glossate.primary" in GLOSSATE_THEME.styles


class TestGetHeaderPanel:
    def test_renders_without_exception(self) -> None:
        panel = get_header_panel()
        _render(panel)

    def test_contains_ascii_logo(self) -> None:
        panel = get_header_panel()
        output = _render(panel)
        # ANSI Shadow block chars that appear in the GLOSSATE logo
        assert "███" in output

    def test_contains_subtitle(self) -> None:
        panel = get_header_panel()
        output = _render(panel)
        assert "GLOSSATE" in output

    def test_returns_panel(self) -> None:
        assert isinstance(get_header_panel(), Panel)


class TestInfoPanel:
    def test_returns_panel(self) -> None:
        result = info_panel("System", {"Device": "mps"})
        assert isinstance(result, Panel)

    def test_renders_without_exception(self) -> None:
        panel = info_panel("System", {"Device": "mps"})
        _render(panel)

    def test_contains_key(self) -> None:
        panel = info_panel("System", {"Device": "mps"})
        output = _render(panel)
        assert "Device" in output

    def test_contains_value(self) -> None:
        panel = info_panel("System", {"Device": "mps"})
        output = _render(panel)
        assert "mps" in output


class TestLogoSmall:
    def test_small_logo_is_str(self) -> None:
        assert isinstance(GLOSSATE_LOGO_SMALL, str)

    def test_small_logo_not_empty(self) -> None:
        assert GLOSSATE_LOGO_SMALL.strip()


class TestGetStyledLogo:
    def test_renders_without_exception(self) -> None:
        logo = get_styled_logo()
        _render(logo)
