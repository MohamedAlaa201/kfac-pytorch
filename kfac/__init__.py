"""Top-level module for K-FAC."""
from __future__ import annotations

import sys

import kfac.assignment as assignment
import kfac.base_preconditioner as base_preconditioner
import kfac.distributed as distributed
import kfac.enums as enums
import kfac.gpt_neox as gpt_neox
import kfac.layers as layers
import kfac.preconditioner as preconditioner
import kfac.scheduler as scheduler
import kfac.tracing as tracing
import kfac.warnings as warnings

if sys.version_info >= (3, 8):  # pragma: >=3.8 cover
    import importlib.metadata as importlib_metadata
else:  # pragma: <3.8 cover
    import importlib_metadata


__version__ = importlib_metadata.version('kfac-pytorch')
