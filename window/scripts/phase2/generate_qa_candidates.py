# This script will:
# Read each PDF from your corpus
# Send the full document text to Gemini with a prompt asking for Q&A pairs
# Parse the response into structured question-answer pairs with supporting spans
# Save results to data/golden/raw_candidates.jsonl


import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import sys
import random

# Add project root directory to sys.path to allow importing ingest.parse
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.parse import extract_pdf_with_page_tiers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
PDF_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/golden")
OUTPUT_FILE = OUTPUT_DIR / "raw_candidates.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 1.5 Flash for its generous free tier limits
MODEL_NAME = "models/gemini-3.5-flash"

def get_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF using your tiered parser."""
    page_texts, _ = extract_pdf_with_page_tiers(pdf_path)
    # Join all pages with clear page markers
    full_text = "\n\n".join([
        f"[Page {i+1}]\n{text}" 
        for i, text in enumerate(page_texts) if text.strip()
    ])
    return full_text

def build_qa_prompt(document_text: str, num_questions: int = 5) -> str:
    return f"""
You are an expert analyst creating a high-quality Q&A dataset from a regulatory document.

DOCUMENT:
{document_text}

TASK:
Generate exactly {num_questions} diverse, high-quality question-answer pairs based SOLELY on the document.

REQUIREMENTS:
1. Questions must be answerable from the document.
2. Answers must be concise and directly supported by the document.
3. supporting_quote must contain the EXACT supporting sentence(s) from the document.
4. Use diverse question types:
   - factoid
   - exact_identifier
   - multi_hop
   - comparative
   - temporal

Return ONLY valid JSON in exactly this structure:

{{
  "qa_pairs": [
    {{
      "question": "string",
      "answer": "string",
      "supporting_quote": "exact quote from document",
      "question_type": "factoid",
      "source_location": "Page X, Section Y"
    }}
  ]
}}

Do not wrap the JSON in markdown code fences.
Do not include explanations outside the JSON.
"""

def generate_qa_for_pdf(pdf_path: Path, num_questions: int = 5) -> list[dict]:
    """Generate Q&A pairs for a single PDF using Gemini."""
    try:
        # Extract text
        logger.info(f"Extracting text from {pdf_path.name}")
        document_text = get_pdf_text(pdf_path)
        
        if len(document_text.strip()) < 100:
            logger.warning(f"{pdf_path.name} has very little text. Skipping.")
            return []
        
        # Truncate if needed (Gemini has 1M token limit)
        # Rough estimate: 1 token ≈ 4 chars, 1M tokens ≈ 4M chars
        # Most RBI circulars are well under this limit
        if len(document_text) > 3_500_000:  # safety margin
            logger.warning(f"{pdf_path.name} exceeds safe length. Truncating.")
            document_text = document_text[:3_500_000]
        
        # Build prompt
        prompt = build_qa_prompt(document_text, num_questions)
        
        # Call Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # Lower temperature for more factual outputs
                "top_p": 0.9,
            }
        )
        
        # Parse response
        response_text = response.text
        # Extract JSON from the response (Gemini may wrap it in markdown)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        qa_pairs = result.get("qa_pairs", [])
        
        # Add source document info to each pair
        for pair in qa_pairs:
            pair["source_pdf"] = pdf_path.name
            pair["source_pdf_path"] = str(pdf_path)
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {pdf_path.name}")
        return qa_pairs
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {pdf_path.name}: {e}")
        logger.debug(f"Response was: {response_text[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return []

def process_all_pdfs(pdfs: list[Path], questions_per_pdf: int = 5):
    """Process all PDFs with rate limit handling."""
    all_candidates = []
    total_pdfs = len(pdfs)
    
    for i, pdf_path in enumerate(pdfs):
        logger.info(f"Processing {i+1}/{total_pdfs}: {pdf_path.name}")
        
        # Generate Q&A
        qa_pairs = generate_qa_for_pdf(pdf_path, questions_per_pdf)
        all_candidates.extend(qa_pairs)
        
        # Save incrementally
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        # Rate limit: wait between requests
        # 15 RPM = 4 seconds between requests minimum
        # Add jitter to avoid hitting limits exactly
        delay = 4 + random.uniform(0.5, 2.0)
        logger.info(f"Waiting {delay:.1f}s before next PDF...")
        time.sleep(delay)
    
    return all_candidates

def main():
    # Get all PDFs (limit to a subset for initial testing)
    pdfs = list(PDF_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs")
    
    # For initial test, process a small batch
    # test_pdfs = pdfs[:5]  # uncomment to test with 5 PDFs first
    
    # Process all
    candidates = process_all_pdfs(pdfs, questions_per_pdf=5)
    
    # Summary
    print("\n" + "="*50)
    print("Q&A CANDIDATE GENERATION SUMMARY")
    print("="*50)
    print(f"PDFs processed: {len(pdfs)}")
    print(f"Total candidates generated: {len(candidates)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()