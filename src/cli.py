"""
Milestone 20: CLI Interface
Command-line interface for the Insurance Enrollment Agent.
"""
import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.predict_tool import predict_employee_enrollment
from tools.rank_tool import rank_employees
from tools.explain_tool import explain_prediction
from tools.refusal_guardrails import check_guardrails

def main():
    parser = argparse.ArgumentParser(
        description="Insurance Enrollment Agent CLI — Predict, Rank, and Explain Employee Benefit Enrollment"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available agent commands")

    # Command 1: Predict
    predict_parser = subparsers.add_parser("predict", help="Predict enrollment probability for an employee ID or custom CSV")
    predict_parser.add_argument("employee_id", type=int, nargs="?", default=None, help="Employee ID (e.g., 12324)")
    predict_parser.add_argument("--data_path", type=str, default=None, help="Path to custom raw employees CSV file")

    # Command 2: Rank
    rank_parser = subparsers.add_parser("rank", help="Rank employees by enrollment probability")
    rank_parser.add_argument("--top_k", type=int, default=10, help="Number of employees to return (default: 10)")
    rank_parser.add_argument("--region", type=str, choices=["Midwest", "Northeast", "South", "West"], help="Filter by region")
    rank_parser.add_argument("--employment_type", type=str, choices=["Full-time", "Part-time", "Contract"], help="Filter by employment type")
    rank_parser.add_argument("--ascending", action="store_true", help="Rank lowest probability first if set")
    rank_parser.add_argument("--capacity", action="store_true", help="Apply regional HR outreach capacity cap")
    rank_parser.add_argument("--data_path", type=str, default=None, help="Path to custom raw employees CSV file")

    # Command 3: Explain
    explain_parser = subparsers.add_parser("explain", help="Explain prediction drivers for an employee ID")
    explain_parser.add_argument("employee_id", type=int, help="Employee ID (e.g., 17825)")
    explain_parser.add_argument("--data_path", type=str, default=None, help="Path to custom raw employees CSV file")

    # Command 4: Query (Natural language prompt with guardrails)
    query_parser = subparsers.add_parser("query", help="Ask a natural language query with safety guardrails")
    query_parser.add_argument("prompt", type=str, help="Natural language query string")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "predict":
        if args.employee_id is not None:
            res = predict_employee_enrollment(employee_id=args.employee_id, data_path=args.data_path)
        elif args.data_path is not None:
            # Predict entire custom CSV
            res = predict_employee_enrollment(data_path=args.data_path)
        else:
            predict_parser.print_help()
            sys.exit(1)
        print(json.dumps(res, indent=2))

    elif args.command == "rank":
        res = rank_employees(
            top_k=args.top_k,
            ascending=args.ascending,
            region=args.region,
            employment_type=args.employment_type,
            apply_hr_capacity_limit=args.capacity,
            data_path=args.data_path
        )
        print(json.dumps(res, indent=2))

    elif args.command == "explain":
        res = explain_prediction(employee_id=args.employee_id)
        print(json.dumps(res, indent=2))

    elif args.command == "query":
        # Check guardrails first
        guard_res = check_guardrails(args.prompt)
        if not guard_res['allowed']:
            print(f"\n[GUARDRAIL REFUSAL] {guard_res['refusal_message']}")
            sys.exit(0)

        print(f"\n[QUERY PROCESSING] Guardrails passed for query: '{args.prompt}'")
        print("Dispatching request to agent tools...")
        # Simple sample dispatch logic
        if "rank" in args.prompt.lower() or "top" in args.prompt.lower():
            res = rank_employees(top_k=5)
            print(json.dumps(res, indent=2))
        else:
            print("Query processed successfully.")

if __name__ == "__main__":
    main()
