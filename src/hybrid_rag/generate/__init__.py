"""Generation: passages -> a grounded, cited answer.

Architecture layers 15-18. Built in Phase 4.
  prompt.py   -- prompt assembly; ordering matters (lost-in-the-middle)
  llm.py      -- LiteLLM call with Groq -> Gemini -> OpenRouter fallback
  validate.py -- Pydantic output shape + every citation span checked against source
  abstain.py  -- score threshold that makes the system say "I do not know""""
