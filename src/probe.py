"""Package entry point for the Environment Agency archive probe.

The evidence-backed implementation remains at the repository root so the
original probe/report pair stays directly runnable. This wrapper gives the
scaffold the documented ``python -m src.probe`` entry point without copying or
silently changing the probe logic.
"""

from probe_ea_hydrology import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
