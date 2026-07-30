# Insurance Enrollment Agent

An end-to-end Machine Learning system and intelligent Outreach Assistant Agent for predicting, ranking, and explaining employee benefit enrollment decisions.

## CLI Usage

```bash
# Predict enrollment probability for an employee
python src/cli.py predict 12324

# Rank top candidates by enrollment probability (with regional filtering and HR capacity cap)
python src/cli.py rank --top_k 5 --region West --capacity

# Explain prediction drivers for an employee
python src/cli.py explain 17825

# Ask a natural language query with safety guardrails
python src/cli.py query "Which plan treats diabetes best?"
```

## Optional: Streamlit Dashboard

To launch the Streamlit frontend dashboard:

```bash
streamlit run src/app.py
```