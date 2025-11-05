import os
import pandas as pd

# Folder where all 30 CSV files are stored
folder_path = "30stocks2015"  # ⬅️ Change this to your folder path

# Columns to keep (case-insensitive)
columns_to_keep = ['time', 'open', 'high', 'low', 'close', 'volume']

# Create a folder for cleaned files
output_folder = os.path.join(folder_path, "ohlcv_data")
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Keep only required columns (case-insensitive)
        df_filtered = df[[col for col in df.columns if col.lower() in columns_to_keep]]

        # Save the filtered file
        output_path = os.path.join(output_folder, filename)
        df_filtered.to_csv(output_path, index=False)

        print(f"✅ Cleaned and saved: {filename}")

print("\n🎉 All files processed and saved in:", output_folder)
