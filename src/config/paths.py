from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ERA5_DIR ="/home/sangonvi/Cefet/repositories/atmoseer/data/reanalisys/cds/era5/pressure"
RADAR_CACHE_DIR = PROJECT_ROOT / "radar_cache"