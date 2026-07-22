"""Papermill engine that executes notebooks inside a Tenki Sandbox microVM."""

import json
import logging
import os
import shlex
import tempfile
from pathlib import Path

import nbformat
from papermill.engines import Engine
from papermill.exceptions import PapermillException
from tenki_sandbox import Sandbox

logger = logging.getLogger("papermill.tenki")

REMOTE_DIR = "/home/tenki/papermill"
REMOTE_INPUT = f"{REMOTE_DIR}/input.ipynb"
REMOTE_OUTPUT = f"{REMOTE_DIR}/output.ipynb"

# Minimal toolchain needed to run nbconvert remotely. The default Tenki image
# (Ubuntu 24.04) ships python3 without pip/ensurepip, so pip itself may need to
# be apt-installed first; PIP_BREAK_SYSTEM_PACKAGES covers PEP 668 images.
BOOTSTRAP_CMD = (
    "set -e; "  # fail fast so an early apt/pip error isn't masked by a later success
    "command -v pip3 >/dev/null 2>&1 "
    "|| (sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip); "
    "PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet "
    "nbconvert nbclient ipykernel jupyter-core"
)

# Probe for the toolchain via imports (PATH-independent).
TOOLCHAIN_PROBE = "python3 -c 'import nbconvert, nbclient, ipykernel'"

# Emits {"kernelspecs": {name: ...}} like `jupyter kernelspec list --json`.
KERNELSPEC_CMD = (
    "python3 -c 'import json; from jupyter_client.kernelspec import KernelSpecManager; "
    'print(json.dumps({"kernelspecs": KernelSpecManager().find_kernel_specs()}))\''
)

# Extra slack on top of execution_timeout for kernel startup + notebook I/O.
EXEC_TIMEOUT_SLACK = 120

# Sandbox lifetime backstop: even if the local process is killed and teardown
# never runs, the VM self-terminates instead of leaking. Derived from
# execution_timeout when known (plus overhead for provisioning + bootstrap),
# otherwise a fixed default.
MAX_DURATION_OVERHEAD = 1200
DEFAULT_MAX_DURATION = 3600
# Bound provisioning so create() can't hang forever waiting for RUNNING.
DEFAULT_CREATE_TIMEOUT = 300


class TenkiRemoteExecutionError(PapermillException):
    """Remote notebook execution inside the sandbox failed."""


class TenkiEngine(Engine):
    """
    Executes the (already parameterized) notebook in a disposable Tenki sandbox.

    Flow: serialize `nb_man.nb` -> upload -> `jupyter nbconvert --execute`
    remotely (a neutral executor, so papermill's parameter cell is not injected
    twice) -> download -> copy executed cells back into `nb_man.nb`.

    Configuration (engine kwargs override environment variables):
        project_id   / TENKI_PROJECT_ID  -- Tenki project to create the sandbox in (required)
        image        / TENKI_IMAGE       -- sandbox image (optional)
        sandbox_name                     -- sandbox display name (default "papermill")
        bootstrap    / TENKI_BOOTSTRAP   -- pip-install the jupyter toolchain in the
                                            sandbox when missing (default enabled;
                                            set TENKI_BOOTSTRAP=0 to disable)
        cpu_cores, memory_mb             -- sandbox resources (optional)
        max_duration                     -- sandbox lifetime backstop in seconds; a
                                            crashed local process can't orphan the VM
                                            (default derived from execution_timeout)
        create_timeout                   -- seconds to wait for the sandbox to reach
                                            RUNNING before giving up (default 300)
    """

    @classmethod
    def execute_managed_notebook(
        cls,
        nb_man,
        kernel_name,
        log_output=False,
        start_timeout=60,
        execution_timeout=None,
        project_id=None,
        image=None,
        sandbox_name="papermill",
        bootstrap=None,
        cpu_cores=None,
        memory_mb=None,
        max_duration=None,
        create_timeout=None,
        **kwargs,
    ):
        project_id = project_id or os.environ.get("TENKI_PROJECT_ID")
        if not project_id:
            raise PapermillException(
                "Tenki project id is required: pass engine kwarg 'project_id' "
                "or set TENKI_PROJECT_ID"
            )
        image = image or os.environ.get("TENKI_IMAGE")
        if bootstrap is None:
            bootstrap = os.environ.get("TENKI_BOOTSTRAP", "1") not in ("0", "false", "no")

        if max_duration is None:
            max_duration = (
                execution_timeout + MAX_DURATION_OVERHEAD
                if execution_timeout
                else DEFAULT_MAX_DURATION
            )

        create_kwargs = {
            "name": sandbox_name,
            "project_id": project_id,
            "max_duration": max_duration,
            "timeout": create_timeout or DEFAULT_CREATE_TIMEOUT,
        }
        if image:
            create_kwargs["image"] = image
        if cpu_cores:
            create_kwargs["cpu_cores"] = cpu_cores
        if memory_mb:
            create_kwargs["memory_mb"] = memory_mb

        with tempfile.TemporaryDirectory() as tmp:
            local_in = Path(tmp) / "input.ipynb"
            local_out = Path(tmp) / "output.ipynb"
            nbformat.write(nb_man.nb, str(local_in))

            logger.info("creating tenki sandbox (project_id=%s image=%s)", project_id, image)
            with Sandbox.create(**create_kwargs) as sb:
                cls._ensure_jupyter(sb, bootstrap)
                kernel = cls._resolve_kernel(sb, kernel_name)

                sb.exec("bash", "-lc", f"mkdir -p {REMOTE_DIR}")
                sb.fs.upload(str(local_in), REMOTE_INPUT)

                logger.info("executing notebook remotely (kernel=%s)", kernel)
                res = sb.exec(
                    "bash",
                    "-lc",
                    cls._nbconvert_cmd(kernel, start_timeout, execution_timeout),
                    **cls._exec_timeout(execution_timeout),
                )
                if log_output and res.stdout_text.strip():
                    logger.info("remote stdout:\n%s", res.stdout_text)

                downloaded = cls._try_download(sb, local_out)
                if not res.ok and not downloaded:
                    raise TenkiRemoteExecutionError(
                        f"remote notebook execution failed ({cls._describe(res)}):\n"
                        f"{res.stderr_text[-4000:]}"
                    )

            executed = nbformat.read(str(local_out), as_version=4)

        # Merge results back; papermill owns writing the output file.
        nb_man.nb.cells = executed.cells
        if "language_info" in executed.metadata:
            nb_man.nb.metadata["language_info"] = executed.metadata["language_info"]
        nb_man.save()

    @classmethod
    def _ensure_jupyter(cls, sb, bootstrap):
        """Verify the sandbox can run nbconvert, installing the toolchain if allowed."""
        if sb.exec("bash", "-lc", TOOLCHAIN_PROBE).ok:
            return
        if not bootstrap:
            raise PapermillException(
                "sandbox image lacks the jupyter toolchain and bootstrap is disabled; "
                "use an image with nbconvert+ipykernel or enable bootstrap"
            )
        logger.info("bootstrapping jupyter toolchain in sandbox")
        res = sb.exec("bash", "-lc", BOOTSTRAP_CMD, timeout=600)
        if not res.ok or not sb.exec("bash", "-lc", TOOLCHAIN_PROBE).ok:
            raise PapermillException(
                f"failed to bootstrap jupyter toolchain in sandbox ({cls._describe(res)}):\n"
                f"{res.stderr_text[-2000:]}"
            )

    @classmethod
    def _resolve_kernel(cls, sb, kernel_name):
        """Map the requested kernel to one available in the sandbox."""
        res = sb.exec("bash", "-lc", KERNELSPEC_CMD)
        available = set()
        if res.ok:
            try:
                available = set(json.loads(res.stdout_text).get("kernelspecs", {}))
            except (json.JSONDecodeError, ValueError):
                available = set()

        # Requested kernel exists remotely: use it as-is.
        if not kernel_name or kernel_name in available:
            return kernel_name
        # Fall back to python3 when the exact kernel is missing but python3 exists.
        if "python3" in available:
            logger.warning("kernel '%s' not in sandbox, falling back to python3", kernel_name)
            return "python3"
        # No usable match: let nbconvert try (and fail loudly with its own error).
        return kernel_name

    @classmethod
    def _nbconvert_cmd(cls, kernel_name, start_timeout, execution_timeout):
        cmd = (
            "python3 -m nbconvert --to notebook --execute "
            f"--ExecutePreprocessor.startup_timeout={int(start_timeout)} "
        )
        if execution_timeout:
            cmd += f"--ExecutePreprocessor.timeout={int(execution_timeout)} "
        if kernel_name:
            cmd += f"--ExecutePreprocessor.kernel_name={shlex.quote(kernel_name)} "
        cmd += f"--output output.ipynb --output-dir {REMOTE_DIR} {REMOTE_INPUT}"
        return cmd

    @classmethod
    def _exec_timeout(cls, execution_timeout):
        """Bound the remote exec call only when the notebook itself is bounded."""
        if execution_timeout:
            return {"timeout": execution_timeout + EXEC_TIMEOUT_SLACK}
        return {}

    @classmethod
    def _try_download(cls, sb, local_out):
        """Fetch the executed notebook; missing output (failed run) is not an error here."""
        try:
            sb.fs.download(REMOTE_OUTPUT, str(local_out))
            return local_out.exists() and local_out.stat().st_size > 0
        except Exception:
            return False

    @classmethod
    def _describe(cls, res):
        """Failure diagnostics beyond exit code: a signal or reason is why an
        exit_code of 0 can still be a failure (e.g. SIGKILL)."""
        parts = [f"exit={res.exit_code}"]
        if res.signal:
            parts.append(f"signal={res.signal}")
        if res.reason:
            parts.append(f"reason={res.reason}")
        if res.errno:
            parts.append(f"errno={res.errno}")
        return ", ".join(parts)
