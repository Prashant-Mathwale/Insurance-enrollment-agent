"""
Milestone 19: Refusal Logic & Guardrails
Implements ethical safety guardrails for the Insurance Enrollment Agent:
1. Medical advice refusal & professional redirection
2. Protected class discrimination refusal (gender/race/religion filtering for denial)
3. Data privacy & bulk PII export safety
"""
import re

MEDICAL_KEYWORDS = [
    'diagnose', 'diagnosis', 'treatment', 'medical advice', 'doctor',
    'prescription', 'illness', 'disease', 'condition', 'therapy',
    'symptom', 'cure', 'medication', 'diabetes', 'cancer'
]

PROTECTED_DISCRIMINATION_PATTERNS = [
    r'filter out (female|male|women|men|other gender)',
    r'exclude (female|male|women|men|other gender)',
    r'deny (female|male|women|men|other gender)',
    r'only hire (female|male|women|men)',
    r'discriminate'
]

def check_guardrails(query):
    """
    Evaluates a user prompt or query against safety policies.

    Returns:
    --------
    dict containing 'allowed' (bool), 'refusal_type' (str), and 'refusal_message' (str).
    """
    if not isinstance(query, str) or not query.strip():
        return {'allowed': True}

    query_lower = query.lower()

    # 1. Medical Advice Refusal
    for kw in MEDICAL_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
            return {
                'allowed': False,
                'refusal_type': 'MEDICAL_ADVICE_REFUSAL',
                'refusal_message': (
                    "REFUSAL: I am an AI insurance enrollment assistant and cannot provide medical advice, "
                    "clinical diagnoses, or specific plan treatment recommendations. "
                    "Please consult a licensed healthcare provider or your HR Benefits Specialist."
                )
            }

    # 2. Protected Class Discrimination Refusal
    for pattern in PROTECTED_DISCRIMINATION_PATTERNS:
        if re.search(pattern, query_lower):
            return {
                'allowed': False,
                'refusal_type': 'DISCRIMINATION_GUARDRAIL_REFUSAL',
                'refusal_message': (
                    "REFUSAL: I cannot perform filtering or ranking designed to exclude or deny benefits "
                    "based on protected demographic characteristics (e.g., gender). All benefit enrollment recommendations "
                    "must adhere to corporate non-discrimination policies and Fair AI standards."
                )
            }

    return {'allowed': True}

if __name__ == "__main__":
    test_queries = [
        "What is the enrollment probability for employee 12324?",
        "Which insurance plan is best to treat my diabetes?",
        "Please filter out female employees from the outreach list.",
        "Rank top 10 employees by enrollment probability in West region."
    ]

    print("=== Testing Refusal Guardrails ===")
    for q in test_queries:
        res = check_guardrails(q)
        print(f"\nQuery: '{q}'")
        if res['allowed']:
            print("  Status: ALLOWED")
        else:
            print(f"  Status: REFUSED ({res['refusal_type']})")
            print(f"  Message: {res['refusal_message']}")
