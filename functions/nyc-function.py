import coiled
from dask.distributed import print
import pandas as pd
import s3fs

s3 = s3fs.S3FileSystem()
filenames = s3.ls("s3://nyc-tlc/trip data/")
filenames = [
    "s3://" + fn
    for fn in filenames
    if "yellow_tripdata_2022" in fn
]

@coiled.run(
    region="us-east-1",
    memory="8 GiB",
)
def process(filename):
    df = pd.read_parquet(filename)
    df = df[df.tip_amount != 0]

    outfile = "s3://oss-shared-scratch/mrocklin/" + filename.split("/")[-1]
    df.to_parquet(outfile)
    print("Finished", outfile)


print(f"\nProcessing {len(filenames)} files")
for filename in filenames:
    print("Processing", filename)
    process(filename)
