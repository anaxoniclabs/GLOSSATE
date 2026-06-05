# SPDX-License-Identifier: MIT
from unittest.mock import patch

import pytest

from glossate.utils.device import get_memory_info, get_optimal_device


class TestGetOptimalDevice:
    def test_returns_mps_when_mps_available(self) -> None:
        with patch("glossate.utils.device._mps_available", return_value=True):
            assert get_optimal_device() == "mps"

    def test_returns_cuda_when_cuda_available_and_mps_not(self) -> None:
        with (
            patch("glossate.utils.device._mps_available", return_value=False),
            patch("glossate.utils.device._cuda_available", return_value=True),
        ):
            assert get_optimal_device() == "cuda"

    def test_returns_cpu_when_neither_available(self) -> None:
        with (
            patch("glossate.utils.device._mps_available", return_value=False),
            patch("glossate.utils.device._cuda_available", return_value=False),
        ):
            assert get_optimal_device() == "cpu"

    def test_override_cpu_always_returns_cpu(self) -> None:
        assert get_optimal_device("cpu") == "cpu"

    def test_override_mps_raises_when_mps_unavailable(self) -> None:
        with patch("glossate.utils.device._mps_available", return_value=False):
            with pytest.raises(ValueError, match="mps"):
                get_optimal_device("mps")

    def test_override_cuda_raises_when_cuda_unavailable(self) -> None:
        with patch("glossate.utils.device._cuda_available", return_value=False):
            with pytest.raises(ValueError, match="cuda"):
                get_optimal_device("cuda")

    def test_override_mps_succeeds_when_mps_available(self) -> None:
        with patch("glossate.utils.device._mps_available", return_value=True):
            assert get_optimal_device("mps") == "mps"

    def test_override_cuda_succeeds_when_cuda_available(self) -> None:
        with patch("glossate.utils.device._cuda_available", return_value=True):
            assert get_optimal_device("cuda") == "cuda"

    def test_unknown_override_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_optimal_device("tpu")

    def test_auto_treated_as_no_override(self) -> None:
        with patch("glossate.utils.device._mps_available", return_value=True):
            assert get_optimal_device("auto") == "mps"


class TestGetMemoryInfo:
    _VM_STAT_OUTPUT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                              100.\n"
        "Pages speculative:                        50.\n"
        "Pages active:                            200.\n"
        "Pages inactive:                          150.\n"
        "Pages wired down:                         75.\n"
    )

    def _mock_run(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.stdout = self._VM_STAT_OUTPUT
        return result

    def test_returns_free_gb_key_on_success(self) -> None:
        with (
            patch("glossate.utils.device.sys.platform", "darwin"),
            patch("glossate.utils.device._cuda_available", return_value=False),
            patch("subprocess.run", return_value=self._mock_run()),
            patch("os.sysconf", return_value=16384),
        ):
            info = get_memory_info()
        assert "free_gb" in info
        assert "active_gb" in info
        assert "inactive_gb" in info
        assert "wired_gb" in info

    def test_free_gb_is_float(self) -> None:
        with (
            patch("glossate.utils.device.sys.platform", "darwin"),
            patch("glossate.utils.device._cuda_available", return_value=False),
            patch("subprocess.run", return_value=self._mock_run()),
            patch("os.sysconf", return_value=16384),
        ):
            info = get_memory_info()
        assert isinstance(info["free_gb"], float)

    def test_free_gb_calculation(self) -> None:
        page_size = 16384
        free_pages = 100
        speculative_pages = 50
        output = (
            "Mach Virtual Memory Statistics:\n"
            f"Pages free:                              {free_pages}.\n"
            f"Pages speculative:                        {speculative_pages}.\n"
            "Pages active:                            0.\n"
            "Pages inactive:                          0.\n"
            "Pages wired down:                        0.\n"
        )
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.stdout = output
        with (
            patch("glossate.utils.device.sys.platform", "darwin"),
            patch("glossate.utils.device._cuda_available", return_value=False),
            patch("subprocess.run", return_value=mock_result),
            patch("os.sysconf", return_value=page_size),
        ):
            info = get_memory_info()
        expected = ((free_pages + speculative_pages) * page_size) / (1024**3)
        assert info["free_gb"] == pytest.approx(expected)

    def test_returns_error_dict_when_subprocess_raises(self) -> None:
        with (
            patch("glossate.utils.device.sys.platform", "darwin"),
            patch("glossate.utils.device._cuda_available", return_value=False),
            patch("subprocess.run", side_effect=Exception("vm_stat failed")),
        ):
            info = get_memory_info()
        assert "error" in info
        assert "free_gb" not in info

    def test_error_dict_contains_message(self) -> None:
        with (
            patch("glossate.utils.device.sys.platform", "darwin"),
            patch("glossate.utils.device._cuda_available", return_value=False),
            patch("subprocess.run", side_effect=Exception("boom")),
        ):
            info = get_memory_info()
        assert "boom" in info["error"]

    def test_cuda_memory_info_preferred_for_cuda_device(self) -> None:
        with patch(
            "glossate.utils.device._cuda_memory_info",
            return_value={"device": "cuda", "free_gb": 1.0, "total_gb": 2.0},
        ):
            info = get_memory_info("cuda")
        assert info["device"] == "cuda"
        assert info["free_gb"] == 1.0
