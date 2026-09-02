# Create src/ingest/chunking.py and write a function that uses LangChain’s RecursiveCharacterTextSplitter with tiktoken as the length function 
# this ensures that chunk_size=512 means 512 tokens, not 512 characters.

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_naive_splitter(chunk_size: int = 512, overlap_tokens: int = 51):
    """
    Returns a RecursiveCharacterTextSplitter that splits by token count.
    overlap_tokens is 10% of 512 = 51 (we round to nearest int).
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokeniser; safe proxy for BGE/E5
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],  # respects paragraphs, sentences, words
        keep_separator=False,
    )
    return splitter

def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Splits a single page's text into chunks, attaching the metadata to each chunk.
    Returns a list of dicts: {"text": str, "meta": dict}
    """
    splitter = get_naive_splitter()
    chunks = splitter.split_text(text)
    
    chunk_records = []
    for i, chunk in enumerate(chunks):
        record = {
            "text": chunk,
            "meta": {
                **metadata,  # pdf_name, page_number, etc. from ChunkMeta
                "chunk_index": i,
                "total_chunks_in_doc": len(chunks),
            }
        }
        chunk_records.append(record)
    return chunk_records