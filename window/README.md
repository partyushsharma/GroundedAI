# Hybrid-RAG Document Parsing & Ingestion Engine

A hybrid document parsing, chunking, and Retrieval-Augmented Generation (RAG) pipeline designed for processing RBI circulars and complex financial/legal PDF documents.

---

## 🏗️ Multi-Tier Ingestion Architecture

The ingestion pipeline (`ingest/parse.py`) uses a 3-tiered fallback strategy per page to maximize extraction speed while ensuring 100% layout and text capture accuracy:

1. **Tier 1 — PyMuPDF (`fitz`)**: Fast text extraction for clean, native digital PDFs.
2. **Tier 2 — Docling**: Deep layout-aware extraction for complex document structures, multi-column layouts, and tables.
3. **Tier 3 — Tesseract OCR**: Full optical character recognition (OCR) fallback for scanned images or pages where text extraction yields poor quality or fails.

---

## 📦 Installed Packages, Dependencies & Rationale

Below is the list of key packages in the environment, their direct dependencies, and why they were added to the project:

### 1. Document Parsing & Layout Engine
- **`pymupdf` (`fitz`)**
  - **Why Added**: Primary Tier 1 parser for high-speed text extraction from digital PDFs.
  - **Key Dependencies**: C-extension bindings for MuPDF engine.
- **`docling` & `docling-core`**
  - **Why Added**: Tier 2 layout-aware parser for complex tables and multi-column documents. Exports structured text to Markdown per page.
  - **Key Dependencies**: `docling-parse`, `docling-ibm-models`, `transformers`, `torch`, `huggingface_hub`, `pydantic`.
- **`onnxruntime`**
  - **Why Added**: Provides accelerated execution for RapidOCR model inference inside Docling, eliminating CPU fallback warnings and accelerating OCR/table detection.
  - **Key Dependencies**: `flatbuffers`, `coloredlogs`, `protobuf`, `humanfriendly`, `sympy`.

### 2. OCR & Image Fallbacks
- **`pytesseract`**
  - **Why Added**: Python wrapper for Tesseract OCR (Tier 3 fallback) when page text extraction fails or image quality is poor.
  - **Key Dependencies**: `Pillow`, Tesseract OCR system binary (`tesseract.exe`).
- **`pdf2image`**
  - **Why Added**: Converts PDF pages into PIL Image objects for input into Tesseract OCR.
  - **Key Dependencies**: `Pillow`, Poppler binary (`pdftoppm.exe`).
- **`pillow` (`PIL`)**
  - **Why Added**: Core image processing library required by `pdf2image`, `pytesseract`, and `docling`.

### 3. Chunking & Tokenization
- **`tiktoken`**
  - **Why Added**: Fast BPE tokenizer (`cl100k_base`) used for exact token count estimation during page chunking.
  - **Key Dependencies**: `regex`, `requests`.

### 4. Web Scraping & Utilities
- **`beautifulsoup4`**
  - **Why Added**: HTML parsing engine for scraping RBI circulars from web pages.
  - **Key Dependencies**: `soupsieve`.
- **`python-dotenv`**
  - **Why Added**: Automatically loads environment configuration and API keys from `.env` file.

### 5. Testing & Environment Doctor
- **`pytest`**
  - **Why Added**: Test framework used for automated validation of unit tests in `tests/`.
  - **Key Dependencies**: `iniconfig`, `pluggy`, `tomli`.
- **`torch` & `transformers`**
  - **Why Added**: Machine learning backend models for Docling table structure and layout detection.

---

## ⚙️ System Binaries

- **Tesseract OCR (`tesseract.exe`)**:
  - Download from: [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
  - Auto-detected at: `C:\Program Files\Tesseract-OCR\tesseract.exe` (or system `PATH`).
- **Poppler (`pdftoppm.exe`)**:
  - Download from: [Poppler Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases)
  - Auto-detected at: `D:\Software\poppler\poppler-26.02.0\Library\bin` (or system `PATH`).

---

## 🛠️ Usage & Verification Commands

### 1. Environment Health Check
Check installed packages, binary paths, and environment configuration:
```bash
python doctor.py
```

### 2. Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### 3. Run Quality Logging Script
Processes all PDFs in `data/raw/` and records page-tier statistics:
```bash
python scripts/build_quality_log.py
```

### 4. Run Chunking Script
Extracts per-page text and generates chunks saved to `data/processed/chunks.jsonl`:
```bash
python scripts/run_naive_chunking.py
```

### 5. Package Management
All currently installed dependencies are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```