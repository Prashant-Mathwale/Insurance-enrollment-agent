import fitz # PyMuPDF

doc = fitz.open("Ml Intern Assignment.pdf")
text = ""
for idx, page in enumerate(doc):
    text += f"\n--- PAGE {idx + 1} ---\n" + page.get_text()

with open("scratch/pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(text)} characters from PDF to scratch/pdf_text.txt")
