import sys


DEBUG = True


def log(message: str) -> None:
    if DEBUG:
        print(f"[APP LOG] {message}", file=sys.stderr)


