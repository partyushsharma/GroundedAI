TECH STACK USED :
- RBI circular Scrap : beautifulsoup4
- Parse all text from PDF : PyMuPDF
- Table format /Small sentenace which not parsed by PyMuPDF (Shift to): Docling
- pytest for testing (Testing command: python -m pytest tests/test_parse.py -v )
- pytesseract – Python wrapper for Tesseract.
- pdf2image – converts PDF pages to PIL images.
- pillow – image handling (already a dependency of pdf2image, but good to have explicitly).