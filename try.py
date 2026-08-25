import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="aqi_karachi_features",
    version=1
)

query = fg.select_all()

df = fg.select_all().show(16752)
df = df.sort_values("timestamp_utc")

print(df.head())
print(df.tail())
