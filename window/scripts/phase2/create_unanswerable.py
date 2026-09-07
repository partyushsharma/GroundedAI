import json
from pathlib import Path

# Your questions from Gemini
questions_from_gemini = [
    "What is the maximum limit for inclusion of general provisions and loss reserves in Tier 2 capital under Basel III norms for scheduled commercial banks?",
    "Are bank exposures to qualified Central Counterparties (QCCPs) exempt from the Liquidity Coverage Ratio (LCR) run-off factor calculations?",
    "Can a scheduled commercial bank reckon its excess SLR securities towards meeting the Net Stable Funding Ratio (NSFR) requirements?",
    "What is the prescribed risk weight for unrated commercial real estate exposures under the standardized approach for capital adequacy?",
    "Under what specific macroeconomic conditions can a bank reduce its Capital Conservation Buffer (CCB) below the prescribed 2.5% without invoking regulatory penalties?",
    "How should a bank treat outstanding credit card receivables for the calculation of the exposure measure under the Basel III Leverage Ratio?",
    "In the case of infrastructure project loans under implementation, what is the maximum permissible extension of the Date of Commencement of Commercial Operations (DCCO) before the asset is classified as sub-standard?",
    "What is the provisioning requirement for the unsecured portion of a doubtful asset that has remained in the 'Doubtful-3' category for more than three years?",
    "Under the RBI's Prudential Framework for Resolution of Stressed Assets, what is the minimum additional provisioning required if a viable restructuring plan is not implemented within 180 days from the end of the review period?",
    "Can a bank upgrade an NPA account to 'standard' category upon partial payment of overdue interest, without clearing the entire principal arrears?",
    "How should a bank classify an infrastructure loan facility where the concession agreement is terminated by the government authority and arbitration is pending?",
    "What are the reporting timelines for a bank to declare an account as a 'Wilful Defaulter' to credit information companies after the Review Committee's decision is finalized?"
]

# Additional questions to reach 20 (you need 8 more)
additional_questions = [
    "What is the maximum tenure for a rupee-denominated bond issued by an Indian company to overseas investors under the External Commercial Borrowings (ECB) framework?",
    "Are co-operative banks required to maintain the same Cash Reserve Ratio (CRR) as scheduled commercial banks?",
    "Can a non-banking financial company (NBFC) issue demand drafts and traveler's cheques without obtaining prior RBI approval?",
    "What is the penalty for failure to appoint a Chief Compliance Officer by the prescribed deadline under the new corporate governance guidelines?",
    "How should a bank calculate the Net Open Position (NOP) limit for forex trading when holding multiple currency pairs?",
    "What is the maximum stake a foreign portfolio investor can hold in a commodity exchange registered in India?",
    "Are payment aggregators required to maintain a separate escrow account for each merchant under the new guidelines?",
    "How many consecutive quarters of profit does a bank need to show before it can declare a dividend after a period of losses?"
]

all_questions = questions_from_gemini + additional_questions

# Create JSONL file
OUTPUT_FILE = Path("data/golden/unanswerable.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Notes explaining why each question is unanswerable
# (You can customize these based on your actual corpus)
base_notes = [
    "Corpus contains Basel III general norms but no specific limit for provisions/loss reserves.",
    "QCCP exemptions are not mentioned in any circular; LCR run-off factors are addressed generically.",
    "NSFR guidelines are not covered in this corpus; only LCR is mentioned.",
    "Risk weights for unrated CRE exposures are not specified in available circulars.",
    "CCB invocation conditions are not detailed in any RBI circular in the corpus.",
    "Credit card receivables treatment under Leverage Ratio is not discussed.",
    "DCCO extensions for infrastructure loans are mentioned but with different thresholds.",
    "Doubtful-3 provisioning requirements are stated but unsecured portion handling is not clear.",
    "Restructuring plan implementation timelines are mentioned but provisioning details differ.",
    "NPA upgrade conditions are specified but partial payment is not addressed.",
    "Infrastructure loan classification during arbitration is not covered.",
    "Wilful Defaulter reporting timelines are not specified in available circulars.",
    "ECB tenure limits for rupee-denominated bonds are not in this corpus.",
    "Co-operative bank CRR requirements are not covered; only scheduled commercial banks.",
    "NBFC issuance of demand drafts is not mentioned in any circular.",
    "Chief Compliance Officer appointment penalties are not specified.",
    "NOP limit calculations for forex trading are not detailed.",
    "FPI stake limits in commodity exchanges are not covered.",
    "Payment aggregator escrow requirements are not in this corpus.",
    "Dividend declaration conditions for banks are mentioned but with different criteria."
]

# Write to JSONL
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for idx, (question, note) in enumerate(zip(all_questions, base_notes)):
        entry = {
            "question": question,
            "answer": "",  # Empty because unanswerable
            "supporting_quote": "",  # Empty because no quote exists
            "source_pdf": "",  # Empty because no source
            "source_pdf_path": "",  # Empty because no source
            "question_type": "unanswerable",
            "gold_chunk_ids": [],  # Empty list
            "is_unanswerable": True,
            "review_status": "Accepted",
            "reviewer_notes": note,
            "id": idx
        }
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"✅ Created {len(all_questions)} unanswerable questions in {OUTPUT_FILE}")
print(f"   {len(questions_from_gemini)} from Gemini + {len(additional_questions)} additional = {len(all_questions)} total")