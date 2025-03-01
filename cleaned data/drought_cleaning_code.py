import pandas as pd


file_path = "LA_drought_raw.csv" 
df = pd.read_csv(file_path)

# Step 2: Convert the 'DATE' column to a usable format (extract the year)
df["YEAR"] = df["DATE"].str.extract(r'(\d{4})').astype(float)  # Extracting year from the date

# Step 3: Filter rows for years 2002-2025
df_filtered = df[(df["YEAR"] >= 2002) & (df["YEAR"] <= 2025)]

# Step 4: Drop the helper 'YEAR' column (if not needed)
df_filtered = df_filtered.drop(columns=["YEAR"])


df_filtered.to_csv(file_path, index=False)

print("CSV file successfully updated with dates from 2002 to 2025!")
import pandas as pd


file_path = "LA_drought_raw.csv"  
df = pd.read_csv(file_path)

# Step 2: Rename columns
df.rename(columns={'DATE': 'Date'}, inplace=True)

# Step 3: Convert `Date` column to datetime format
df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace("d_", ""), format='%Y%m%d')

# Step 4: Handle missing values (replace NaN with 0 or appropriate method)
df.fillna(0, inplace=True)

# Step 5: Remove invalid or unnecessary columns
if '-9' in df.columns:
    df.drop(columns=['-9'], inplace=True)

# Step 6: Convert numerical columns to float (excluding 'Date' column)
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 7: Sort data chronologically
df.sort_values(by='Date', inplace=True)

# Step 8: Overwrite the original file
df.to_csv(file_path, index=False)

print(f"Data cleaning completed! The file '{file_path}' has been updated.")
import pandas as pd


file_path = "LA_drought_raw.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove rows with invalid dates
df = df.dropna(subset=["Date"])

# Remove duplicate rows if any
df = df.drop_duplicates()

# Fill or remove NaN values in numerical columns
df.fillna(0, inplace=True)

# Save the cleaned data back to the same file
df.to_csv(file_path, index=False)

print("Data cleaning complete. File has been updated.")
