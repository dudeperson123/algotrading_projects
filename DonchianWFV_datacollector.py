import os
import zipfile
import pandas as pd
import re

folder_path = "BTCUSDT_BINANCEUS_1H"

combined_file = os.path.join(folder_path, "BTCUSDT_BINANCEUS_1H_COMBINED.csv")
formatted_file = os.path.join(folder_path, "BTCUSDT_1H.csv")


for filename in os.listdir(folder_path):
    if filename.lower().endswith(".zip"):
        zip_path = os.path.join(folder_path, filename)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)

            print(f"Extracted: {filename}")

            # Delete zip after extraction
            os.remove(zip_path)
            print(f"Deleted: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

csv_files = []

for filename in os.listdir(folder_path):

    if filename in [
        "BTCUSDT_BINANCEUS_1H_COMBINED.csv",
        "BTCUSDT_1H.csv"
    ]:
        continue

    if filename.lower().endswith(".csv"):

        # Extract YYYY-MM from filename
        match = re.search(r'(\d{4})-(\d{2})', filename)

        if match:
            year = int(match.group(1))
            month = int(match.group(2))

            csv_files.append((year, month, filename))

# Sort chronologically
csv_files.sort()

print("\nCSV files in chronological order:")
for _, _, f in csv_files:
    print(f)

dfs = []

for _, _, filename in csv_files:
    file_path = os.path.join(folder_path, filename)

    try:
        df = pd.read_csv(file_path)

        dfs.append(df)

        print(f"Loaded: {filename}")

    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Merge all data
combined_df = pd.concat(dfs, ignore_index=True)

# Sort chronologically by open_time
combined_df = combined_df.sort_values("open_time")

# Save raw combined version
combined_df.to_csv(combined_file, index=False)

print(f"\nSaved raw combined CSV:")
print(combined_file)


# Convert timestaps
combined_df["datetime"] = pd.to_datetime(
    combined_df["open_time"],
    unit="ms"
)

# Keep only desired columns
final_df = combined_df[
    ["datetime", "open", "high", "low", "close", "volume"]
].copy()

# Format datetime exactly like BTCUSDT_1H.csv
final_df["datetime"] = final_df["datetime"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Save final formatted file
final_df.to_csv(formatted_file, index=False)

print(f"\nSaved formatted CSV:")
print(formatted_file)

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Keep only the final formatted file
        if filename != "BTCUSDT_1H.csv":
            try:
                os.remove(file_path)
                print(f"Deleted CSV: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")

print("\nDone.")
