# papermill-tenki

A [papermill](https://github.com/nteract/papermill) engine that executes
parameterized Jupyter notebooks inside disposable
[Tenki Sandbox](https://tenki.cloud/docs/sandbox/quick-start-sandbox) microVMs.

Parameter injection happens locally in papermill as usual; the already
parameterized notebook is uploaded to a fresh sandbox, executed there with
`jupyter nbconvert --execute`, and the executed notebook (outputs, plots,
errors) is downloaded and saved to papermill's output path. The sandbox is
destroyed afterwards.

## Install

```bash
pip install papermill-tenki
```

## Setup

```bash
export TENKI_API_KEY=tk_...          # from your Tenki workspace settings
export TENKI_PROJECT_ID=...          # project to create sandboxes in
```

## Usage

CLI (`-p name value` injects a notebook parameter, only if the notebook has a
`parameters` cell that declares it):

```bash
papermill input.ipynb output.ipynb --engine tenki -p learning_rate 0.5
```

Python:

```python
import papermill

papermill.execute_notebook(
    "input.ipynb",
    "output.ipynb",
    engine_name="tenki",
    # notebook parameters — only for notebooks with a `parameters` cell; omit otherwise:
    parameters={"learning_rate": 0.5},
    # optional engine kwargs (override env vars); see Configuration below:
    # project_id="...", image="...", sandbox_name="my-run",
    # cpu_cores=4, memory_mb=8192,
)
```

`bootstrap` is enabled by default and installs the jupyter toolchain in the
sandbox for you — only set `bootstrap=False` if your `image` already has it.

## Configuration

| Engine kwarg   | Env var            | Default     | Meaning                                        |
| -------------- | ------------------ | ----------- | ---------------------------------------------- |
| `project_id`   | `TENKI_PROJECT_ID` | — (required)| Tenki project the sandbox is created in        |
| `image`        | `TENKI_IMAGE`      | Tenki default | Sandbox image                                |
| `sandbox_name` | —                  | `papermill` | Sandbox display name                           |
| `bootstrap`    | `TENKI_BOOTSTRAP`  | enabled     | pip-install nbconvert/ipykernel when missing   |
| `cpu_cores`    | —                  | Tenki default | Sandbox CPU cores                            |
| `memory_mb`    | —                  | Tenki default | Sandbox memory                               |

`TENKI_API_KEY` is read by the Tenki SDK itself.

Notebook dependencies (numpy, pandas, ...) must exist in the sandbox: bake
them into a custom `image`, or add a bootstrap cell at the top of the notebook.
The kernel requested by the notebook must be installed in the sandbox; when it
isn't and `python3` is available, the engine falls back to `python3` with a
warning.

## Timeouts

papermill's `--start-timeout` and `--execution-timeout` are forwarded to the
remote `nbconvert` run (`startup_timeout` / `timeout` on the
`ExecutePreprocessor`).

## Tests

```bash
pip install -e .[dev]
pytest                       # unit tests, fully mocked, no network
TENKI_API_KEY=... TENKI_PROJECT_ID=... pytest tests/test_live_smoke.py   # live smoke test
```
