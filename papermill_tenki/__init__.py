"""Papermill engine for Tenki Sandbox."""

# Import papermill first so its engine registration (which loads this package's
# entry point) completes before we bind the engine here — avoids a circular
# import when papermill_tenki is imported before papermill.
import papermill.engines  # noqa: F401

from .engine import TenkiEngine, TenkiRemoteExecutionError

__version__ = "0.2.0"
__all__ = ["TenkiEngine", "TenkiRemoteExecutionError"]
