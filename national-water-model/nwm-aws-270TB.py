"""
This example was adapted from https://github.com/dcherian/dask-demo/blob/main/nwm-aws.ipynb
"""

import coiled
import dask
import flox.xarray
import fsspec
import numpy as np
import rioxarray
import xarray as xr


# optionally run with coiled run
# coiled run --region us-east-1 --vm-type m6g.xlarge python nwm-aws.py

cluster = coiled.Cluster(
    name="nwm-1979-2020",
    region="us-east-1", # close to data
    n_workers=10,
    scheduler_vm_types="r7g.xlarge", # ARM instance
    worker_vm_types="r7g.2xlarge",
    compute_purchase_option="spot_with_fallback" # use spot, replace with on-demand
)

client = cluster.get_client()
client.restart()
cluster.adapt(minimum=10, maximum=200)

ds = xr.open_zarr(
    fsspec.get_mapper(
        "s3://noaa-nwm-retrospective-2-1-zarr-pds/rtout.zarr",
        anon=True
    ),
    consolidated=True,
    chunks={"time": 896, "x": 350, "y": 350}
)

subset = ds.zwattablrt.sel(
    time=slice("1979-02-01", "2020-12-31")
)

fs = fsspec.filesystem("s3", requester_pays=True)

with dask.annotate(retries=3):
    counties = rioxarray.open_rasterio(
        fs.open("s3://nwm-250m-us-counties/Counties_on_250m_grid.tif"),
        chunks="auto"
    ).squeeze()

# remove any small floating point error in coordinate locations
_, counties_aligned = xr.align(subset, counties, join="override")

counties_aligned = counties_aligned.persist()

county_id = np.unique(counties_aligned.data).compute()
county_id = county_id[county_id != 0]
print(f"There are {len(county_id)} counties!")

county_mean = flox.xarray.xarray_reduce(
    subset,
    counties_aligned.rename("county"),
    func="mean",
    expected_groups=(county_id,),
)

county_mean.load()
yearly_mean = county_mean.mean("time")
# optionally, save dataset for further analysis
# print("Saving")
# yearly_mean.to_netcdf("mean_zwattablrt_nwm_1979_2020.nc")
cluster.shutdown()
