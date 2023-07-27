import pandas as pd

df = pd.read_parquet(
    "s3://nyc-tlc/trip data/yellow_tripdata_2023-04.parquet"
)

print("Head")
print("====")
print(df.head())

print("Columns")
print("=======")
print(df.columns)

print("Tip Percentage")
print("==============")
print((df.tip_amount != 0).mean())

print("Uploading tipped rides to S3")


df = df[df.tip_amount != 0]
df.to_parquet(
    "s3://oss-shared-scratch/mrocklin/nyc-tipped-2023-04.parquet"
)


print("Done")
