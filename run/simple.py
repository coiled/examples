import coiled
import pandas as pd

df = pd.read_csv("data.csv")

@coiled.function(
    region="us-west-1",
)
def process(df):
    df["z"] = df.x + df.y
    return df

print(process(df))
