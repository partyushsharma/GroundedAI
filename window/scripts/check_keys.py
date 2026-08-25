import os
from dotenv import load_dotenv
load_dotenv()
 
for name in ["GROQ_API_KEY", "GEMINI_API_KEY"]:
    v = os.getenv(name)
    print(f"{name}: {v[:8] + '...' if v else 'MISSING'}")
print()
 
from litellm import completion
 
CANDIDATES = [
    ("groq/openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("groq/openai/gpt-oss-20b",  {"reasoning_effort": "low"}),
    ("gemini/gemini-3.6-flash",  {}),
]
 
working = []
for model, extra in CANDIDATES:
    try:
        r = completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=300,          # reasoning models need headroom
            **extra,
        )
        msg = r.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None)
        u = r.usage
 
        status = content if content else "(empty)"
        print(f"{model:30s} -> {status}")
        print(f"{'':30s}    tokens: {u.completion_tokens} out / {u.prompt_tokens} in"
              + (f", reasoning: {len(reasoning)} chars" if reasoning else ""))
        if content:
            working.append(model)
    except Exception as e:
        print(f"{model:30s} -> FAILED: {type(e).__name__}: {str(e)[:120]}")