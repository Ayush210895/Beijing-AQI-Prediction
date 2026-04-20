"""Convenience wrapper for local training runs.

Run with:
    PYTHONPATH=src python scripts/train.py --data-dir data/PRSA_Data_20130301-20170228
"""

from beijing_aqi.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
