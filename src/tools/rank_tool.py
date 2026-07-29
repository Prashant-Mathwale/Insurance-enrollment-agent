"""
Milestone 17: Ranking Tool
Agent tool for ranking employees by predicted enrollment probability.
Supports top-K ranking, regional HR capacity constraints, and filtering.
"""
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from predict import predict

def rank_employees(
    top_k=10,
    ascending=False,
    region=None,
    employment_type=None,
    apply_hr_capacity_limit=False,
    data_path=None
):
    """
    Ranks employees based on predicted enrollment probability.

    Parameters:
    -----------
    top_k : int, default=10
        Number of top employees to return.
    ascending : bool, default=False
        If False, ranks highest probability first. If True, ranks lowest probability first.
    region : str, optional
        Filter predictions by specific region ('Midwest', 'Northeast', 'South', 'West').
    employment_type : str, optional
        Filter by employment type ('Full-time', 'Part-time', 'Contract').
    apply_hr_capacity_limit : bool, default=False
        If True, caps the returned count by the region's hr_outreach_capacity.
    data_path : str, optional
        Path to raw employees CSV.

    Returns:
    --------
    dict containing ranking summary, parameters used, and ranked employee list.
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "../../data/employees_raw.csv")

    if not os.path.exists(data_path):
        return {'status': 'error', 'message': f"Data file not found at {data_path}"}

    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.drop_duplicates(subset=['employee_id'], keep=False)

    # Apply region / employment type filters prior to inference
    filtered = df_raw.copy()
    if region is not None:
        filtered = filtered[filtered['region'].str.lower() == region.lower()]
        if filtered.empty:
            return {'status': 'error', 'message': f"No employees found matching region '{region}'."}

    if employment_type is not None:
        filtered = filtered[filtered['employment_type'].str.lower() == employment_type.lower()]
        if filtered.empty:
            return {'status': 'error', 'message': f"No employees found matching employment_type '{employment_type}'."}

    res_df = predict(filtered)

    # Merge predictions back to raw info for rich output
    merged = filtered.merge(res_df[['employee_id', 'predicted_probability', 'predicted_class']], on='employee_id')
    merged = merged.sort_values(by='predicted_probability', ascending=ascending)

    # Check HR capacity constraint if requested
    capacity_cap = None
    if apply_hr_capacity_limit and region is not None:
        regions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/region_benefit_profiles.csv")
        df_reg = pd.read_csv(regions_path)
        match_reg = df_reg[df_reg['region'].str.lower() == region.lower()]
        if not match_reg.empty:
            capacity_cap = int(match_reg.iloc[0]['hr_outreach_capacity'])

    effective_k = top_k
    if capacity_cap is not None:
        effective_k = min(top_k, capacity_cap)

    ranked_subset = merged.head(effective_k)

    rankings = []
    for rank_idx, (_, row) in enumerate(ranked_subset.iterrows(), start=1):
        rec = {
            'rank': rank_idx,
            'employee_id': int(row['employee_id']),
            'age': int(row['age']) if pd.notnull(row['age']) else None,
            'salary': float(row['salary']) if pd.notnull(row['salary']) else None,
            'region': str(row['region']),
            'employment_type': str(row['employment_type']),
            'plan_tier_requested': str(row['plan_tier_requested']) if pd.notnull(row['plan_tier_requested']) else 'Unknown',
            'enrollment_probability': round(float(row['predicted_probability']), 4),
            'predicted_enrolled': int(row['predicted_class'])
        }
        rankings.append(rec)

    return {
        'status': 'success',
        'total_eligible': len(merged),
        'returned_count': len(rankings),
        'hr_capacity_cap_applied': capacity_cap,
        'filters': {
            'region': region,
            'employment_type': employment_type,
            'ascending': ascending
        },
        'rankings': rankings
    }

if __name__ == "__main__":
    print("=== Testing Ranking Tool: Top 5 Highest Probability ===")
    top5 = rank_employees(top_k=5, ascending=False)
    print(json.dumps(top5, indent=2))

    print("\n=== Testing Ranking Tool: West Region with HR Capacity Cap ===")
    west_rank = rank_employees(top_k=5, region='West', apply_hr_capacity_limit=True)
    print(json.dumps(west_rank, indent=2))
