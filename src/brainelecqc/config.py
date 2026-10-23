"""Storage configuration resolved per application; no source-repository defaults."""
import os
from pathlib import Path


class Settings:
    @property
    def storage_root(self):
        # Flask is imported only by the optional HTTP adapter or an active caller.
        try:
            from flask import current_app, has_app_context
            override = current_app.config.get("STORAGE_ROOT") if has_app_context() else None
        except ImportError:
            override = None
        return Path(override or os.environ.get("BRAINELECQC_STORAGE_ROOT") or
                    Path.home() / ".local" / "share" / "brainelecqc").expanduser().resolve()

    @property
    def db_path(self):
        return self.storage_root / "brainelecqc.sqlite3"

    @property
    def inputs_root(self):
        return self.storage_root / "inputs"

    @property
    def derivatives_root(self):
        return self.storage_root / "derivatives"


settings = Settings()
