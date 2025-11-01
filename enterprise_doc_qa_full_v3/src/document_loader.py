import os
from pathlib import Path
def load_file(path):
    p = Path(path)
    if p.suffix.lower() == '.pdf':
        try:
            import fitz
            doc = fitz.open(str(p))
            text = ''
            for page in doc:
                text += page.get_text()
            return text
        except Exception:
            return p.read_text(encoding='utf-8', errors='ignore')
    elif p.suffix.lower() == '.docx':
        try:
            from docx import Document
            doc = Document(str(p))
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception:
            return p.read_text(encoding='utf-8', errors='ignore')
    else:
        return p.read_text(encoding='utf-8', errors='ignore')
