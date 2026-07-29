import os
import pandas as pd
import numpy as np

def explore_datasets():
    # Resolve path relative to script directory to allow running from any CWD
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    regions_path = os.path.join(current_dir, "../data/region_benefit_profiles.csv")
    
    print(f"Loading datasets...\n - Employees: {employees_path}\n - Regions: {regions_path}\n")
    df_emp = pd.read_csv(employees_path)
    df_reg = pd.read_csv(regions_path)
    
    # 1. Shapes
    print("=== Dataset Shapes ===")
    print(f"Employees shape: {df_emp.shape}")
    print(f"Regions shape: {df_reg.shape}\n")
    
    # 2. Columns & Data Types
    print("=== Employees Columns & Data Types ===")
    print(df_emp.dtypes)
    print("\n=== Regions Columns & Data Types ===")
    print(df_reg.dtypes)
    print("")
    
    # 3. Missing Values
    print("=== Employees Missing Values ===")
    print(df_emp.isnull().sum())
    print("\n=== Regions Missing Values ===")
    print(df_reg.isnull().sum())
    print("")
    
    # 4. Basic Range/Value Statistics
    print("=== Employees Numeric Column Summary ===")
    print(df_emp.describe())
    print("\n=== Regions Numeric Column Summary ===")
    print(df_reg.describe())
    print("")
    
    # 5. Categorical Cardinality
    print("=== Employees Categorical Cardinality & Unique Values ===")
    categorical_cols = df_emp.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df_emp[col].nunique()
        print(f"Column '{col}' has {unique_count} unique values. Top 5:")
        print(df_emp[col].value_counts(dropna=False).head(5))
        print("-" * 30)

if __name__ == "__main__":
    explore_datasets()
