"""Allow `python -m phiserve`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
