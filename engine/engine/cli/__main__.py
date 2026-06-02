"""Entry for ``python -m engine.cli`` (driver.py spawns this)."""
import sys

from . import main

if __name__ == "__main__":
    main()
    sys.exit(0)
