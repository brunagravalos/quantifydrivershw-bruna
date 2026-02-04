import xarray as xr
from pathlib import Path
import zarr

def nc_to_zarr(nc_path, zarr_path, chunks=None):
    ds = xr.open_dataset(nc_path, chunks=chunks)
    ds.to_zarr(zarr_path, mode="w", consolidated=True)

# Example usage
base_nc = Path("/gpfs/scratch/bsc32/bsc214253/mockdata")
base_zarr = Path("/gpfs/scratch/bsc32/bsc214253/climate_data_new.zarr")

# ERA5
nc_to_zarr(
    base_nc / "mockLargeScale_data" / "file_g500.nc",
    base_zarr / "era5" / "g500",
    chunks={"time": 100}
)

nc_to_zarr(
    base_nc / "mockLargeScale_data" / "file_g200.nc",
    base_zarr / "era5" / "g200",
    chunks={"time": 100}
)

nc_to_zarr(
    base_nc / "mockLargeScale_data" / "file_psl.nc",
    base_zarr / "era5" / "psl",
    chunks={"time": 100}
)

# CO2
nc_to_zarr(
    Path("/gpfs/scratch/bsc32/bsc167965/data/daily_co2_JJA_standardized.nc"),
    base_zarr / "aux" / "co2",
    chunks={"time": 500}
)

# ERA5-Land per site & percentile
for site in ["cordoba", "hannover", "stockholm","belgrado","marrakech", "lyon"]:
    for perc in ["90p"]:
        nc_to_zarr(
            base_nc / "mockLocalScale_data" / f"file_local_{perc}_{site}.nc",
            base_zarr / "era5land" / perc / site,
            chunks={"time": 200}
        )


# 1. Initialize the root as a group (mode='a' creates it if missing)
# This creates the hidden .zgroup file at the top level
zarr.open_group(str(base_zarr), mode='a')

# 2. NOW consolidate the metadata
# This scans all children and creates the .zmetadata file at the root
zarr.consolidate_metadata(str(base_zarr))