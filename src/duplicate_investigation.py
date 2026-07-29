import os
import pandas as pd

def investigate_duplicates():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    df = pd.read_csv(employees_path)
    
    # Identify duplicate employee IDs
    dup_mask = df.duplicated(subset=['employee_id'], keep=False)
    duplicates = df[dup_mask].sort_values(by='employee_id')
    
    print(f"Total rows with duplicate employee_id: {len(duplicates)}")
    print(f"Number of unique duplicate IDs: {duplicates['employee_id'].nunique()}\n")
    
    # Investigate if there are differences between the rows of each duplicate ID
    # Check features (excluding the 'enrolled' target)
    feature_cols = [c for c in df.columns if c != 'enrolled']
    
    diff_records = []
    
    for emp_id, group in duplicates.groupby('employee_id'):
        print(f"--- Employee ID: {emp_id} ---")
        print(f"Number of rows: {len(group)}")
        
        # Check if the feature values are identical across the duplicates
        first_row_features = group.iloc[0][feature_cols]
        is_feature_identical = True
        
        for idx in range(1, len(group)):
            other_row_features = group.iloc[idx][feature_cols]
            # Handle NaNs correctly in comparison
            identical_mask = (first_row_features == other_row_features) | (first_row_features.isnull() & other_row_features.isnull())
            if not identical_mask.all():
                is_feature_identical = False
                differing_cols = first_row_features.index[~identical_mask].tolist()
                print(f"WARNING: Feature mismatch in columns: {differing_cols}")
                for col in differing_cols:
                    print(f"  Row 0: {first_row_features[col]}")
                    print(f"  Row {idx}: {other_row_features[col]}")
        
        # Check the labels
        labels = group['enrolled'].tolist()
        is_label_identical = len(set(labels)) == 1
        
        print(f"Features identical: {is_feature_identical}")
        print(f"Labels: {labels} (Identical: {is_label_identical})")
        print("-" * 40)
        
        diff_records.append({
            'employee_id': emp_id,
            'features_identical': is_feature_identical,
            'labels_identical': is_label_identical,
            'labels': labels
        })

if __name__ == "__main__":
    investigate_duplicates()
