import zarr
import numpy as np

dataset_dir = "./datasets/corrdiff/"
root = zarr.open(dataset_dir + "train.zarr", mode="r")

target = root["target"]

print("GLOBAL MIN:", target[:].min())
print("GLOBAL MAX:", target[:].max())
print("GLOBAL MEAN:", target[:].mean())
