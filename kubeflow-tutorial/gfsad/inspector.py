import os
import pandas as pd

def inspect_gfsad_dataset(file_path="gfsad1000_tutorial_input.csv"):
    print("=" * 60)
    print(" 🌾 OPERATION CROPWATCH: DATASET INSPECTION REPORT")
    print("=" * 60)
    
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"[!] Error: Dataset file '{file_path}' not found in current working directory ({os.getcwd()}).")
        print("Please ensure your downloaded GFSAD dataset is saved here.")
        return
    
    # 2. Load the dataset
    df = pd.read_csv(file_path)
    print(f"[+] Successfully loaded: {file_path}")
    print(f"[+] Total Samples (Rows): {len(df)}")
    print(f"[+] Total Features (Columns): {len(df.columns)}")
    print("-" * 60)
    
    # 3. Display Column Names and Data Types
    print("\n--- Column Schema & Types ---")
    print(df.dtypes)
    print("-" * 60)
    
    # 4. Check for Missing / Null Values
    missing_values = df.isnull().sum()
    print("\n--- Missing Values Check ---")
    if missing_values.sum() == 0:
        print("[✓] Clean! No missing values detected in the dataset.")
    else:
        print(missing_values[missing_values > 0])
    print("-" * 60)
    
    # 5. Profile Target Class Distribution (Landcover Classes)
    if 'landcover' in df.columns:
        print("\n--- Landcover Class Distribution (Target) ---")
        class_counts = df['landcover'].value_counts().sort_index()
        class_proportions = df['landcover'].value_counts(normalize=True).sort_index() * 100
        
        dist_df = pd.DataFrame({
            'Sample Count': class_counts,
            'Percentage (%)': class_proportions.round(2)
        })
        print(dist_df)
    else:
        print("\n[!] Warning: 'landcover' target column not found in dataset columns.")
    print("-" * 60)
    
    # 6. Basic Numerical Feature Summary Statistics
    print("\n--- Numerical Features Summary (First few columns) ---")
    feature_cols = [col for col in df.columns if col not in ['landcover', 'latitude', 'longitude']]
    if feature_cols:
        print(df[feature_cols].describe().T[['mean', 'std', 'min', 'max']])
    else:
        print(df.describe().T)
        
    print("=" * 60)
    print(" Inspection complete. Ready for discussion!")
    print("=" * 60)

if __name__ == "__main__":
    inspect_gfsad_dataset()
