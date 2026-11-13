import pandas as pd

dirty = pd.read_csv("data/hospital/dirty.csv")
clean = pd.read_csv("data/hospital/clean.csv")

print("Dirty shape:", dirty.shape)
print("Clean shape:", clean.shape)

print("\nOnly in dirty:")
print(set(dirty.columns) - set(clean.columns))

print("\nOnly in clean:")
print(set(clean.columns) - set(dirty.columns))

print("\nColumn order identical:",
      list(dirty.columns) == list(clean.columns))

print("\nDirty columns:", dirty.columns.tolist())
print("Clean columns:", clean.columns.tolist())