
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator
import os
import pyarrow.dataset as ds
import pyarrow.compute as pc

from src.config.paths import DATA_DIR, ERA5_DIR

import logging
import sys


# =========================================
# LOGGER UTILS DOMAIN
# =========================================

def setup_logger():
    logger = logging.getLogger("corrdiff")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logger()

# =========================================
# ERA5 DOMAIN
# =========================================
class ERA5Dataset:

    def __init__(self, path: str, variables, start_date, end_date, n_threads=8):
        
        self.path = Path("/home/sangonvi/Cefet/repositories/atmoseer/data/reanalysis/cds/era5/pressure")
        self.variables = variables
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_threads = n_threads

        self.df = None
        self.lat = None
        self.lon = None

    def _load_file(self, f):
        try:
            logger.info("Loading ERA5...")
            df = pd.read_parquet(f)
    
            if "time" not in df.columns:
                df = df.reset_index()

            df["time"] = pd.to_datetime(df["valid_time"])
            return df
        except:
            return None

    def load(self):
        logger.info(f"Loading ERA5 from {self.path}")
        
        if not Path(self.path).exists():
            raise ValueError(f"Path não existe: {self.path}")

        # Cria dataset (detecta year= / month= automaticamente)
        dataset = ds.dataset(self.path, format="parquet", partitioning="hive")

        logger.info("Schema:", dataset.schema)

        # =========================================
        # FILTRO TEMPORAL
        # =========================================
        # Descobre coluna de tempo
        time_col = None
        for col in dataset.schema.names:
            if col in ["time", "valid_time"]:
                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found in dataset")

        logger.info(f"Using time column: {time_col}")

        # Converte datas para timestamp
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        # Filtro direto no dataset (super eficiente)
        filter_expr = (
            (pc.field(time_col) >= pc.scalar(start)) &
            (pc.field(time_col) <= pc.scalar(end))
        )

        # Seleciona colunas necessárias
        columns = ["latitude", "longitude", time_col] + self.variables

        # =========================================
        # LEITURA EFICIENTE
        # =========================================
        table = dataset.to_table(
            columns=columns,
            filter=filter_expr
        )

        if table.num_rows == 0:
            logger.error("No data returned after filtering")
            raise ValueError("No data found for the specified date range")

        logger.info(f"Rows loaded: {table.num_rows}")
        df = table.to_pandas()

        # Padroniza nome da coluna
        if time_col != "time":
            df["time"] = df[time_col]

        df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values(["time", "latitude", "longitude"])

        self.df = df
        self.lat = np.sort(df["latitude"].unique())
        self.lon = np.sort(df["longitude"].unique())

        logger.info(f"ERA5 loaded: {df.shape}")

    def interpolate(self, era5_t, target_lat, target_lon):
        grids = []

        for var in self.variables:
            grid = era5_t.pivot(
                index="latitude",
                columns="longitude",
                values=var
            ).values

            interp = RegularGridInterpolator(
                (self.lat, self.lon),
                grid,
                bounds_error=False,
                fill_value=np.nan
            )

            pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=-1)
            vals = interp(pts).reshape(target_lat.shape)

            grids.append(vals)

        return np.stack(grids, axis=0)


# =========================================
# RADAR DOMAIN
# =========================================
import math
import os
import pandas as pd
import numpy as np
from PIL import Image


class ColorInterpolator:
    def __init__(self, legend_colors, legend_values):
        self.legend_colors = legend_colors
        self.legend_values = legend_values

    @staticmethod
    def rgb_distance(c1, c2):
        return math.sqrt(
            (c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2 +
            (c1[2] - c2[2]) ** 2
        )

    def interpolate(self, rgb):
        if rgb == (0, 0, 0):
            return 0

        distances = [self.rgb_distance(rgb, lc) for lc in self.legend_colors]

        min_idx1 = distances.index(min(distances))
        distances[min_idx1] = float("inf")
        min_idx2 = distances.index(min(distances))

        c1, c2 = self.legend_colors[min_idx1], self.legend_colors[min_idx2]
        v1, v2 = self.legend_values[min_idx1], self.legend_values[min_idx2]

        dist_c1_c2 = self.rgb_distance(c1, c2)
        dist_c1_rgb = self.rgb_distance(c1, rgb)

        t = dist_c1_rgb / dist_c1_c2 if dist_c1_c2 != 0 else 0

        return v1 + t * (v2 - v1)


class RadarDataProcessor:
    def __init__(self, radar_path, inicio, fim, frequencia):
        self.radar_path = radar_path
        self.inicio = inicio
        self.fim = fim
        self.frequencia = frequencia

        self.pos_sumare = (-22.955139, -43.248278)

        self.legend_values = [50, 45, 40, 35, 30, 25, 20, 0]
        self.legend_colors = [
            (197, 0, 197),
            (227, 6, 5),
            (255, 112, 0),
            (195, 230, 0),
            (4, 85, 4),
            (19, 122, 19),
            (0, 167, 12),
            (0, 0, 0),
        ]

        self.interpolator = ColorInterpolator(
            self.legend_colors,
            self.legend_values
        )

    def _build_filepath(self, dt):
        file = dt.strftime("%Y_%m_%d_%H_%M.png")
        return os.path.join(
            self.radar_path,
            dt.strftime("%Y"),
            dt.strftime("%m"),
            dt.strftime("%d"),
            file
        )

    def _extract_pixel_value(self, img, latitude, longitude):
        pos_sumare_img = (img.height / 2, img.width / 2)

        dify = pos_sumare_img[0] / self.pos_sumare[0]
        difx = pos_sumare_img[1] / self.pos_sumare[1]

        posx = self.pos_sumare[1] - ((longitude - self.pos_sumare[1]) * 32.5)
        valorx = posx * difx

        posy = self.pos_sumare[0] + ((latitude - self.pos_sumare[0]) * 19.5)
        valory = posy * dify

        rgb_im = img.convert("RGB")
        r, g, b = rgb_im.getpixel((valorx, valory))

        return self.interpolator.interpolate((r, g, b))

    def get_radar_data(self, latitude, longitude):
        latitude = float(latitude)
        longitude = float(longitude)

        data_inicial = self.inicio + " 00:00:00"
        data_final = self.fim + " 23:58:00"

        datas = pd.date_range(
            start=data_inicial,
            end=data_final,
            freq=self.frequencia
        )

        radar_data = pd.DataFrame({"time": datas})
        radar_data["reflect"] = np.nan

        for idx, row in radar_data.iterrows():
            dt = row["time"]
            file_path = self._build_filepath(dt)

            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path)
                    value = self._extract_pixel_value(
                        img,
                        latitude,
                        longitude
                    )
                    radar_data.at[idx, "reflect"] = value
                except Exception as e:
                    logger.info("Erro ao processar imagem:", str(e))

        return radar_data
    
class RadarDataset:

    def __init__(self, cache_dir, resolution_km, lat_range, lon_range, data_fn, n_threads=10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.res_km = resolution_km
        self.data_fn = data_fn
        self.n_threads = n_threads

        self.lat_range = lat_range
        self.lon_range = lon_range

        self.lat = None
        self.lon = None
        self.data = {}
        self.time_map = {}

        self._build_grid()

    def _build_grid(self):
        deg = self.res_km / 111.0

        self.lat = np.arange(self.lat_range[0], self.lat_range[1], deg)
        self.lon = np.arange(self.lon_range[0], self.lon_range[1], deg)

        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)

        self.lat_idx = {round(v, 4): i for i, v in enumerate(self.lat)}
        self.lon_idx = {round(v, 4): j for j, v in enumerate(self.lon)}

    def _cache_path(self, lat, lon):
        return self.cache_dir / f"lat_{lat:.4f}_lon_{lon:.4f}.parquet"

    def _load_or_download(self, lat, lon):
        path = self._cache_path(lat, lon)

        if path.exists():
            try:
                df = pd.read_parquet(path)
                df["time"] = pd.to_datetime(df["time"])
                return (lat, lon, df.set_index("time"))
            except Exception as e:
                logger.warning(f"Error on reading cache ({lat},{lon}): {e}")

        try:
            df = self.data_fn.get_radar_data(latitude=lat, longitude=lon)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
            df = df.resample("1h").max()
            df.reset_index().to_parquet(path)

            return (lat, lon, df)

        except Exception as e:
            logger.error(f"Error on radar data ({lat},{lon}): {e}")
            return (lat, lon, None)

    def build_cache(self):
        tasks = [(lat, lon) for lat in self.lat for lon in self.lon]
        total = len(tasks)
        logger.info(f"Starting radar cache build ({total} points)")
        
        done = 0
        with ThreadPoolExecutor(max_workers=self.n_threads) as ex:
            futures = [ex.submit(self._load_or_download, lat, lon) for lat, lon in tasks]

            for f in as_completed(futures):
                lat, lon, df = f.result()
                self.data[(round(lat,4), round(lon,4))] = df
               
                done += 1
                if done % 50 == 0 or done == total:
                    logger.info(f"Radar progress: {done}/{total}")
        
        logger.info("Radar cache built")

    def build_time_index(self):
        logger.info("Building radar time index")

        count = 0

        for (lat, lon), df in self.data.items():
            if df is None:
                continue

            for t in df.index:
                self.time_map.setdefault(t, []).append(
                    (lat, lon, df.loc[t].values[0])
                )
                count += 1

        logger.info(f"Time index entries: {count}")

    def get_grid(self, t):
        grid = np.full(self.Lat.shape, np.nan)

        if t not in self.time_map:
            return grid

        for lat, lon, val in self.time_map[t]:
            i = self.lat_idx.get(lat)
            j = self.lon_idx.get(lon)

            if i is not None and j is not None:
                grid[i, j] = val

        return grid


# =========================================
# DATASET BUILDER
# =========================================
class CorrDiffDatasetBuilder:

    def __init__(self, era5, radar, output_dir, patch_size=32, stride=16):
        self.era5 = era5
        self.radar = radar
        self.output_dir = Path(output_dir)

        self.patch_size = patch_size
        self.stride = stride

        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)

    def build(self):
        times = sorted(self.era5.df["time"].unique())
        logger.info(f"Total timesteps: {len(times)}")

        sample_id = 0

        for t in times:
            logger.info(f"Processing timestep: {t}")

            era5_t = self.era5.df[self.era5.df["time"] == t]

            if len(era5_t) == 0:
                logger.warning(f"No ERA5 data for {t}")
                continue

            X = self.era5.interpolate(
                era5_t,
                self.radar.Lat,
                self.radar.Lon
            )

            Y = self.radar.get_grid(t)

            if np.isnan(Y).all():
                logger.warning(f"All NaN radar grid at {t}")
                continue

            patches_created = 0

            H, W = Y.shape

            for i in range(0, H - self.patch_size + 1, self.stride):
                for j in range(0, W - self.patch_size + 1, self.stride):

                    xp = X[:, i:i+self.patch_size, j:j+self.patch_size]
                    yp = Y[i:i+self.patch_size, j:j+self.patch_size]

                    valid_ratio = np.mean(~np.isnan(yp))

                    if valid_ratio < 0.05:
                        continue

                    mask = ~np.isnan(yp)
                    yp = np.nan_to_num(yp, nan=0.0)

                    np.savez_compressed(
                        self.output_dir / "train" / f"{sample_id:06d}.npz",
                        input=xp.astype(np.float32),
                        target=yp[None, ...].astype(np.float32),
                        mask=mask[None, ...].astype(np.float32)
                    )

                    sample_id += 1
                    patches_created += 1

            logger.info(f"Patches created at {t}: {patches_created}")

        logger.info(f"Dataset final size: {sample_id}")


# =========================================
# USO
# =========================================
def main():

    era5 = ERA5Dataset(
        path= ERA5_DIR,
        variables=["v", "u"],
        start_date="2024-01-22 00:00:00",
        end_date="2024-01-22 23:00:00",
     )

    era5.load()

    
    radar = RadarDataset(
        cache_dir="/home/sangonvi/Cefet/repositories/atmoseer/radar_cache",
        resolution_km=2,
        lat_range=(-23.5, -22.25),
        lon_range=(-44.0, -42.5),
        data_fn=RadarDataProcessor(radar_path="/home/sangonvi/Cefet/repositories/atmoseer/data/radar_sumare",
                                   inicio="2024-01-22", 
                                   fim="2024-01-22", 
                                   frequencia="1h")
    )

    radar.build_cache()
    radar.build_time_index()

    builder = CorrDiffDatasetBuilder(
        era5,
        radar,
        output_dir="dataset"
    )

    builder.build()


if __name__ == "__main__":
    main()