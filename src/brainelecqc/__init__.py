"""Electrophysiology QC storage and faithful window access."""
__version__ = "0.1.0"


def create_app(config=None):
    """Construct the optional local HTTP adapter, using an independent data store."""
    from flask import Flask
    from .api import ephys_bp, ephys_samples_bp
    from .db import init_db

    app = Flask(__name__)
    app.config.update(config or {})
    app.register_blueprint(ephys_bp)
    app.register_blueprint(ephys_samples_bp)
    with app.app_context():
        init_db()
    return app
