import coiled
import pandas as pd

df = pd.read_csv("run/data.csv")

@coiled.run(
    region="us-west-1",
    cpu=32,
    keepalive="10 minutes",
)
def process(df):
    df["z"] = df.x + df.y
    return df

print(process(df))
