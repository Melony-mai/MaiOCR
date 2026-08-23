"""Unit tests for engine helpers (no model loading required)."""

from pathlib import Path

import pytest
import rapidocr

from maiocr.ocr.engine import (
    KANA_RATIO_THRESHOLD,
    OCREngine,
    OcrResult,
    _forbid_downloads,
    japan_dict_path,
)


def _bare_engine(mode="dml", label="GPU/DirectML", accel=None):
    """OCREngine without model loading (recovery logic only)."""
    eng = object.__new__(OCREngine)
    eng._engine = None
    eng._japan_engine = None
    eng._japan_model_path = None
    eng._gpu_failures = 0
    eng._mode, eng._label = mode, label
    eng._accel_params = dict(accel or {"use_dml": True})
    return eng


class TestGpuRecovery:
    class Flaky:
        def __init__(self, failures):
            self.remaining = failures

        def __call__(self, arr):
            if self.remaining > 0:
                self.remaining -= 1
                raise RuntimeError("device removed")

            class Raw:
                def __init__(self):
                    self.txts = ["ok"]
                    self.scores = [1.0]
                    self.boxes = None

            return Raw()

    def test_transient_gpu_failure_recovers_on_accelerator(self, monkeypatch):
        eng = _bare_engine()
        eng._engine = self.Flaky(1)
        rebuilds = []

        def fake_create():
            rebuilds.append(1)
            eng._engine = self.Flaky(0)
            return eng._engine

        monkeypatch.setattr(eng, "_create_engine", fake_create)

        img = type("Img", (), {"convert": lambda self, m: None})()
        result = eng.recognize(img)

        assert len(rebuilds) == 1          # GPU retried, not abandoned
        assert eng._accel_params           # still on the accelerator
        assert eng._label == "GPU/DirectML"
        assert result.lines == ["ok"]

    def test_persistent_failure_degrades_to_cpu(self, monkeypatch):
        eng = _bare_engine()

        def always_broken():
            return self.Flaky(10**9)  # keeps raising

        eng._engine = always_broken()
        monkeypatch.setattr(eng, "_create_engine", always_broken)

        # Exhaust recovery attempts with direct calls.
        from maiocr.ocr.engine import _MAX_GPU_RECOVERIES

        for i in range(_MAX_GPU_RECOVERIES):
            eng._ensure_cpu_fallback(RuntimeError("still dead"))
            assert eng._accel_params  # GPU kept so far

        eng._ensure_cpu_fallback(RuntimeError("final"))
        assert not eng._accel_params       # now degraded
        assert eng._mode == "cpu"
        assert eng._label == "CPU"

    def test_healthy_run_resets_recovery_budget(self, monkeypatch):
        eng = _bare_engine()
        eng._engine = self.Flaky(0)
        eng.recognize(type("Img", (), {"convert": lambda self, m: None})())
        assert eng._gpu_failures == 0

_looks_japanese = OCREngine._looks_japanese


class TestOcrResult:
    def test_text_joins_lines(self):
        r = OcrResult(lines=["a", "b"], scores=[0.9, 0.8])
        assert r.text == "a\nb"

    def test_mean_score(self):
        assert OcrResult(["a"], [0.5]).mean_score == 0.5
        assert OcrResult([], []).mean_score == 0.0

    def test_empty_text(self):
        assert OcrResult([], []).text == ""


class TestJapaneseHeuristic:
    def test_kana_heavy(self):
        text = "こんにちは世界、テストです。"
        assert _looks_japanese(text)

    def test_chinese_only(self):
        assert not _looks_japanese("这是中文文本，没有假名。")

    def test_english_only(self):
        assert not _looks_japanese("hello world")

    def test_mixed_mostly_chinese(self):
        # ~10% kana among hanzi stays below the 12% threshold.
        text = "中" * 90 + "い" * 10
        assert not _looks_japanese(text)

    def test_threshold_value_sane(self):
        assert 0 < KANA_RATIO_THRESHOLD < 1


class TestJapanModelParams:
    """
    Contract tests: rapidocr's ParseParams.update_batch requires Enum
    instances for ocr_version / model_type (strings raise TypeError).
    """

    def _updated_cfg(self, tmp_path: Path, dict_file=None):
        from rapidocr.utils.parse_parameters import ParseParams
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

        cfg_path = Path(rapidocr.__file__).parent / "config.yaml"
        cfg = ParseParams.load(cfg_path)
        params = {
            "Rec.model_path": str(tmp_path / "japan.onnx"),
            "Rec.lang_type": LangRec.JAPAN,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
        }
        if dict_file is not None:
            params["Rec.rec_keys_path"] = str(dict_file)
        return ParseParams.update_batch(cfg, params)

    def test_rapidocr_accepts_our_params(self, tmp_path):
        cfg = self._updated_cfg(tmp_path)  # must not raise TypeError
        assert str(cfg.Rec.model_path).endswith("japan.onnx")

    def test_dict_path_lookup(self, tmp_path):
        model = tmp_path / "japan_PP-OCRv4_rec_mobile.onnx"
        model.write_bytes(b"x")
        assert japan_dict_path(model) is None

        dict_file = tmp_path / "japan_PP-OCRv4_rec_mobile.txt"
        dict_file.write_text("a\nb\n", encoding="utf-8")
        found = japan_dict_path(model)
        assert found == dict_file

        cfg = self._updated_cfg(tmp_path, dict_file)
        assert str(cfg.Rec.rec_keys_path) == str(dict_file)


class TestOfflineGuard:
    """The offline guard must allow existing files, refuse downloads."""

    @pytest.fixture()
    def downloader(self):
        from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
        from rapidocr.utils.log import logger as _rapid_logger

        class Input(DownloadFileInput):
            def __init__(self, file_url, save_path):
                super().__init__(
                    file_url=file_url, save_path=save_path, logger=_rapid_logger
                )

        calls = []

        def fake_run(cls, input_params):
            calls.append(input_params.file_url)
            if not Path(input_params.save_path).exists():
                raise RuntimeError(f"download attempted: {input_params.file_url}")

        original = DownloadFile.run
        DownloadFile.run = classmethod(fake_run)
        yield DownloadFile, Input, calls
        DownloadFile.run = original

    def test_existing_file_passes(self, downloader, tmp_path):
        dl, inp, calls = downloader
        target = tmp_path / "model.onnx"
        target.write_bytes(b"x")
        with _forbid_downloads():
            dl.run(inp(file_url="http://example.invalid/m", save_path=target))
        assert calls == ["http://example.invalid/m"]

    def test_missing_file_refused(self, downloader, tmp_path):
        dl, inp, _calls = downloader
        with (
            pytest.raises(RuntimeError, match="offline mode"),
            _forbid_downloads(),
        ):
            dl.run(
                inp(
                    file_url="http://example.invalid/m",
                    save_path=tmp_path / "missing.onnx",
                )
            )

    def test_guard_restores_original(self, downloader):
        from rapidocr.utils.download_file import DownloadFile

        before_func = DownloadFile.run.__func__
        with _forbid_downloads():
            pass
        assert DownloadFile.run.__func__ is before_func
