"""Tests for launch-at-login (registry Run key) management."""

import os
import sys
from pathlib import Path

import pytest

from maiocr.utils import autostart


class FakeKey:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    REG_SZ = 1

    def __init__(self):
        self.store = {}
        self.fail_on_set = False

    def OpenKey(self, root, subkey):
        return FakeKey()

    def CreateKey(self, root, subkey):
        return FakeKey()

    def QueryValueEx(self, key, name):
        if name not in self.store:
            raise FileNotFoundError(2, "value not found")
        return self.store[name], 1

    def SetValueEx(self, key, name, reserved, type_, value):
        if self.fail_on_set:
            raise OSError(5, "access denied")
        self.store[name] = value

    def DeleteValue(self, key, name):
        if name not in self.store:
            raise FileNotFoundError(2, "value not found")
        del self.store[name]


@pytest.fixture
def fake_winreg(monkeypatch):
    fake = FakeWinreg()
    monkeypatch.setattr(autostart, "winreg", fake)
    return fake


def test_launch_command_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\MaiOCR\MaiOCR.exe")

    assert autostart.launch_command() == [r"C:\Apps\MaiOCR\MaiOCR.exe"]
    assert autostart.workdir() == Path(r"C:\Apps\MaiOCR")


def test_launch_command_dev(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    cmd = autostart.launch_command()
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("main.py")
    assert os.path.isabs(cmd[1])


def test_command_line_quotes_spaces(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\MaiOCR\MaiOCR.exe")

    assert (
        autostart.command_line()
        == '"C:\\Program Files\\MaiOCR\\MaiOCR.exe"'
    )


@pytest.mark.skipif(sys.platform != "win32", reason="win32 registry API")
def test_enable_disable_roundtrip(fake_winreg):
    assert not autostart.is_enabled()

    line = autostart.set_enabled(True)
    assert fake_winreg.store["MaiOCR"] == line
    assert "MaiOCR" in line
    assert autostart.is_enabled()

    autostart.set_enabled(False)
    assert "MaiOCR" not in fake_winreg.store
    assert not autostart.is_enabled()


@pytest.mark.skipif(sys.platform != "win32", reason="win32 registry API")
def test_disable_when_absent_is_noop(fake_winreg):
    autostart.set_enabled(False)
    assert not autostart.is_enabled()


@pytest.mark.skipif(sys.platform != "win32", reason="win32 registry API")
def test_registry_error_propagates(fake_winreg):
    fake_winreg.fail_on_set = True
    with pytest.raises(OSError):
        autostart.set_enabled(True)


@pytest.mark.skipif(sys.platform != "win32", reason="win32 creation flags")
def test_spawn_relaunch_detached(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\MaiOCR\MaiOCR.exe")

    spawned = []
    monkeypatch.setattr(
        autostart.subprocess,
        "Popen",
        lambda cmd, cwd=None, **kw: spawned.append((cmd, cwd, kw)) or object(),
    )

    autostart.spawn_relaunch()

    assert len(spawned) == 1
    cmd, cwd, kwargs = spawned[0]
    assert cmd == [r"C:\Apps\MaiOCR\MaiOCR.exe"]
    assert str(cwd) == r"C:\Apps\MaiOCR"
    assert kwargs["creationflags"] == (
        __import__("subprocess").DETACHED_PROCESS
        | __import__("subprocess").CREATE_NEW_PROCESS_GROUP
    )
