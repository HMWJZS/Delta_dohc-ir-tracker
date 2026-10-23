"""Locate resources in either the DOHC tree or the standalone wheel."""
from pathlib import Path


def resource_root():
    module = Path(__file__).resolve().parent
    return module.parent if module.name == '_core' else module.parents[1]
