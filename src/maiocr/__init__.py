__all__ = ["main"]


def __getattr__(name):
    # Lazy re-export: keeps `from maiocr import main` working (entry point)
    # without importing the whole GUI/OCR stack for every maiocr.* import.
    if name == "main":
        from maiocr.app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
