"""Tiny logging helper for verbose pipeline progress.

Usage: from logging_utils import log
log("fetching Wikipedia for %s", name)   # only prints when VERBOSE=1
"""
import logging
import os
import sys

_VERBOSE = os.getenv("VERBOSE", "0") in ("1", "true", "yes")

_logger = logging.getLogger("medsumgraph")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG if _VERBOSE else logging.WARNING)


def log(msg: str, *args):
    if _VERBOSE:
        _logger.info(msg, *args)
