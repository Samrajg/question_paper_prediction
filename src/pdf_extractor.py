import fitz  # PyMuPDF
import re
import pandas as pd

class PDFExtractor:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path):
        """
        Extract text from a PDF, page by page.
        Returns a list of dicts with page_number and text.
        """
        pages_content = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                # Basic normalization
                text = self._normalize_text(text)
                pages_content.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return pages_content

    def _normalize_text(self, text):
        """Normalize whitespace and clean up common extraction artifacts."""
        if not text:
            return ""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def parse_questions_from_text(self, pages_content, filename=""):
        """
        Attempt to heuristically detect questions and marks.
        Very basic implementation that looks for "Q[number]." or "[number]." patterns.
        """
        extracted_questions = []
        
        question_pattern = re.compile(r'(?:Q|Question)?\s*(\d+)\.\s*(.*?)(?=(?:Q|Question)?\s*\d+\.|\Z)', re.IGNORECASE | re.DOTALL)
        
        for page in pages_content:
            text = page["text"]
            matches = question_pattern.findall(text)
            for match in matches:
                q_num = match[0]
                q_text = match[1].strip()
                
                # Attempt to extract marks at the end of the question e.g., "[3]" or "(3)"
                marks_pattern = re.compile(r'[\(\[](\d+)[\)\]]\s*$')
                marks_match = marks_pattern.search(q_text)
                marks = None
                if marks_match:
                    marks = float(marks_match.group(1))
                    q_text = marks_pattern.sub('', q_text).strip()
                
                if len(q_text) > 5: # Ignore very short artifacts
                    extracted_questions.append({
                        "question_number": q_num,
                        "question_text": q_text,
                        "marks": marks,
                        "source_page": page["page_number"],
                        "source_file": filename
                    })
                    
        return extracted_questions

    def process_pdf(self, file_path, filename):
        pages = self.extract_text_from_pdf(file_path)
        questions = self.parse_questions_from_text(pages, filename)
        return pd.DataFrame(questions)
