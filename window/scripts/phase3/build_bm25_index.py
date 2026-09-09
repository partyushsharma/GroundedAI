import json
import pickle
from pathlib import Path
import bm25s
from bm25s.tokenization import Tokenizer
import nltk
from nltk.stem import PorterStemmer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")
OUTPUT_INDEX_DIR = Path("data/indexes/bm25_index")
OUTPUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)


class StemmedTokenizer:
    """Tokeniser that lowercases, splits, and optionally stems."""
    def __init__(self, stem=True):
        self.stemmer = PorterStemmer() if stem else None
        self.tokenizer = Tokenizer()
    
    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens


def main():
    # Load chunks
    texts = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk['text'])
    logger.info(f"Loaded {len(texts)} chunks")

    # Tokenize corpus using bm25s tokenizer with stemming and stopwords
    logger.info("Tokenizing corpus...")
    stemmer = PorterStemmer()
    stemmer_fn = lambda tokens: [stemmer.stem(t) for t in tokens]
    corpus_tokens = bm25s.tokenize(texts, stopwords="english", stemmer=stemmer_fn)

    # Build BM25 index
    bm25 = bm25s.BM25(corpus=texts)
    logger.info("Building BM25 index...")
    bm25.index(corpus_tokens)
    logger.info(f"Index built – vocabulary size: {len(bm25.vocab_dict)}")

    # Save index and corpus using bm25s built-in save
    bm25.save(str(OUTPUT_INDEX_DIR))
    
    # Save tokenizer config
    with open(OUTPUT_INDEX_DIR / "tokenizer_config.json", 'w', encoding='utf-8') as f:
        json.dump({"stem": True, "stopwords": "english"}, f, indent=2)
    
    logger.info(f"BM25 index saved to {OUTPUT_INDEX_DIR}")

if __name__ == "__main__":
    main()