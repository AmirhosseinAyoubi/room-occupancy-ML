import pandas as pd

# Load your CSV
df = pd.read_csv("file.csv")

# Insert new column at the beginning
df.insert(0, "index_number", range(len(df)))

# Save back to CSV
df.to_csv("output.csv", index=False)