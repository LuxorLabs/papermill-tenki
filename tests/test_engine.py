"""Unit tests for TenkiEngine with a mocked Sandbox (no network, no API key)."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook, new_output
from papermill.exceptions import PapermillException
from tenki_sandbox import CommandResult

from papermill_tenki import TenkiEngine, TenkiRemoteExecutionError
from papermill_tenki.engine import (
    BOOTSTRAP_CMD,
    KERNELSPEC_CMD,
    REMOTE_INPUT,
    REMOTE_OUTPUT,
    TOOLCHAIN_PROBE,
)


def make_notebook():
    nb = new_notebook(cells=[new_code_cell("x = 1"), new_code_cell("print(x)")])
    nb.metadata["papermill"] = {}  # normally added by papermill's iorw layer
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def result(exit_code=0, stdout=b"", stderr=b""):
    return CommandResult(argv=["bash"], exit_code=exit_code, stdout=stdout, stderr=stderr)


KERNELSPECS = json.dumps({"kernelspecs": {"python3": {}}}).encode()


class FakeSandbox:
    """Scriptable stand-in for a Tenki sandbox: records commands, simulates nbconvert."""

    def __init__(self, jupyter_present=True, kernelspecs=KERNELSPECS, nbconvert=result()):
        self.jupyter_present = jupyter_present
        self.kernelspecs = kernelspecs
        self.nbconvert = nbconvert
        self.commands = []
        self.exec_kwargs = []
        self.uploaded_nb = None
        self.remote_output_written = False
        self.fs = MagicMock()
        self.fs.upload.side_effect = self._upload
        self.fs.download.side_effect = self._download

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def exec(self, *argv, **kwargs):
        cmd = argv[-1]
        self.commands.append(cmd)
        self.exec_kwargs.append(kwargs)
        if cmd == TOOLCHAIN_PROBE:
            return result(0 if self.jupyter_present else 1)
        if cmd == BOOTSTRAP_CMD:
            self.jupyter_present = True  # bootstrap "installs" the toolchain
            return result(0)
        if cmd == KERNELSPEC_CMD:
            return result(0, stdout=self.kernelspecs)
        if "-m nbconvert" in cmd:
            self.remote_output_written = self.nbconvert.exit_code == 0
            return self.nbconvert
        return result(0)

    def _upload(self, local, remote):
        assert remote == REMOTE_INPUT
        self.uploaded_nb = nbformat.read(local, as_version=4)

    def _download(self, remote, local):
        assert remote == REMOTE_OUTPUT
        if not self.remote_output_written:
            raise FileNotFoundError(remote)
        executed = self.uploaded_nb
        for i, cell in enumerate(c for c in executed.cells if c.cell_type == "code"):
            cell.execution_count = i + 1
            cell.outputs = [new_output("stream", name="stdout", text=f"cell-{i}\n")]
        nbformat.write(executed, local)


def run_engine(sandbox, nb=None, kernel_name="python3", **kwargs):
    with patch("papermill_tenki.engine.Sandbox") as sandbox_cls:
        sandbox_cls.create.return_value = sandbox
        out = TenkiEngine.execute_notebook(
            nb or make_notebook(),
            kernel_name,
            progress_bar=False,
            project_id=kwargs.pop("project_id", "proj-123"),
            **kwargs,
        )
    return out, sandbox_cls


def test_success_merges_executed_cells():
    sb = FakeSandbox()
    nb, sandbox_cls = run_engine(sb)

    sandbox_cls.create.assert_called_once_with(name="papermill", project_id="proj-123")
    sb.fs.upload.assert_called_once()
    sb.fs.download.assert_called_once()
    assert [c.outputs[0]["text"] for c in nb.cells] == ["cell-0\n", "cell-1\n"]
    assert all(c.metadata.papermill["status"] == "completed" for c in nb.cells)


def test_nbconvert_command_shape():
    sb = FakeSandbox()
    run_engine(sb, start_timeout=45, execution_timeout=300)

    cmd = next(c for c in sb.commands if "-m nbconvert" in c)
    assert "--ExecutePreprocessor.kernel_name=python3" in cmd
    assert "--ExecutePreprocessor.startup_timeout=45" in cmd
    assert "--ExecutePreprocessor.timeout=300" in cmd
    assert REMOTE_INPUT in cmd
    # exec call is bounded with slack on top of the notebook timeout
    kwargs = sb.exec_kwargs[sb.commands.index(cmd)]
    assert kwargs == {"timeout": 420}


def test_missing_project_id_raises(monkeypatch):
    monkeypatch.delenv("TENKI_PROJECT_ID", raising=False)
    with patch("papermill_tenki.engine.Sandbox"):
        with pytest.raises(PapermillException, match="project id is required"):
            TenkiEngine.execute_notebook(make_notebook(), "python3", progress_bar=False)


def test_project_id_from_env(monkeypatch):
    monkeypatch.setenv("TENKI_PROJECT_ID", "proj-env")
    sb = FakeSandbox()
    with patch("papermill_tenki.engine.Sandbox") as sandbox_cls:
        sandbox_cls.create.return_value = sb
        TenkiEngine.execute_notebook(make_notebook(), "python3", progress_bar=False)
    sandbox_cls.create.assert_called_once_with(name="papermill", project_id="proj-env")


def test_remote_failure_raises_with_stderr():
    sb = FakeSandbox(nbconvert=result(1, stderr=b"CellExecutionError: boom"))
    with pytest.raises(TenkiRemoteExecutionError, match="boom"):
        run_engine(sb)


def test_bootstrap_installs_toolchain_when_missing():
    sb = FakeSandbox(jupyter_present=False)
    run_engine(sb)
    assert BOOTSTRAP_CMD in sb.commands


def test_bootstrap_disabled_raises():
    sb = FakeSandbox(jupyter_present=False)
    with pytest.raises(PapermillException, match="bootstrap is disabled"):
        run_engine(sb, bootstrap=False)


def test_kernel_falls_back_to_python3():
    sb = FakeSandbox()
    run_engine(sb, kernel_name="julia-1.9")
    cmd = next(c for c in sb.commands if "-m nbconvert" in c)
    assert "--ExecutePreprocessor.kernel_name=python3" in cmd


def test_engine_registered_via_entry_point():
    from papermill.engines import papermill_engines

    assert papermill_engines.get_engine("tenki") is TenkiEngine


def test_import_before_papermill_does_not_circular_import():
    # A fresh process where papermill_tenki is imported first must not crash on
    # papermill's eager entry-point registration. Run in a subprocess so the
    # import order is not polluted by this test module's own imports.
    result = subprocess.run(
        [sys.executable, "-c", "import papermill_tenki; papermill_tenki.TenkiEngine"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
