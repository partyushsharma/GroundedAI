"""Evaluation: the ruler everything else is measured against.

Architecture layers 21-26. The golden set is Phase 2; the rest Phase 3-5.
  retrieval_metrics.py  -- recall@k, nDCG, MRR. No model calls, so free (Phase 2)
  generation_metrics.py -- RAGAS faithfulness / relevancy. Costs API calls (Phase 5)
  judge.py              -- LLM judge, validated against hand labels via Cohen kappa
  failure_classifier.py -- retrieval failure vs generation failure (Phase 4)
  ablation.py           -- runs every config and writes the README results table"""
