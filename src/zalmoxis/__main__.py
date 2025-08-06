from __future__ import annotations

from .zalmoxis import load_zalmoxis_config, post_processing

if __name__ == "__main__":
    config_params = load_zalmoxis_config()
    post_processing(config_params)
