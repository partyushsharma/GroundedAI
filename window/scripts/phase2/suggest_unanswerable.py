import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "models/gemini-3.5-flash"

prompt = """
I have a corpus of RBI circulars and Master Directions covering:
- Foreign exchange remittances
- KYC/AML norms
- NBFC regulations
- Payment systems
- Priority sector lending
- Bank branch licensing

Please suggest 30 plausible, realistic regulatory questions that a compliance officer might ask, but which are NOT covered in any of these topics. These will be used as unanswerable questions to test a RAG system's abstention.

Each question should:
- Sound like a legitimate regulatory query
- Be answerable in principle, but not from the given corpus
- Use domain-specific terminology (e.g., "CRR", "SLR", "FEMA", "Basel III")
- Not be too obviously out-of-scope (e.g., no tax or insurance questions)

Output only a numbered list of questions.
"""

model = genai.GenerativeModel(MODEL)
response = model.generate_content(prompt)
print(response.text)