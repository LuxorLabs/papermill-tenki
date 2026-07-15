"""Live smoke test: runs a real notebook in a real Tenki sandbox.

Requires TENKI_API_KEY and TENKI_PROJECT_ID in the environment; skipped otherwise.
"""

import os

import nbformat
import papermill
import pytest
from nbformat.v4 import new_code_cell, new_notebook

pytestmark = pytest.mark.skipif(
    not (os.environ.get("TENKI_API_KEY") and os.environ.get("TENKI_PROJECT_ID")),
    reason="TENKI_API_KEY / TENKI_PROJECT_ID not set",
)


def test_parameterized_notebook_runs_in_sandbox(tmp_path):
    nb = new_notebook(
        cells=[
            new_code_cell("x = 1", metadata={"tags": ["parameters"]}),
            new_code_cell("print(x * 2)"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    )
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    nbformat.write(nb, str(input_path))

    papermill.execute_notebook(
        str(input_path),
        str(output_path),
        engine_name="tenki",
        parameters={"x": 21},
        progress_bar=False,
    )

    out = nbformat.read(str(output_path), as_version=4)
    prints = [
        o["text"]
        for c in out.cells
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    ]
    assert "42\n" in prints
