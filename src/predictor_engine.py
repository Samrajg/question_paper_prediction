import io
import os
import re
import zipfile
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class QuestionPredictorEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def extract_zip(self, zip_file_bytes) -> dict:
        """
        Extracts all PDFs from a zip file.
        Returns a dict of {filename: pdf_bytes}.
        """
        pdf_files = {}
        with zipfile.ZipFile(io.BytesIO(zip_file_bytes)) as z:
            for file_info in z.infolist():
                if file_info.filename.startswith('__MACOSX/') or file_info.filename.endswith('/'):
                    continue
                if file_info.filename.lower().endswith('.pdf'):
                    clean_name = os.path.basename(file_info.filename)
                    if clean_name:
                        pdf_files[clean_name] = z.read(file_info.filename)
        return pdf_files

    def pdf_to_text(self, pdf_bytes: bytes) -> str:
        """Extracts text from a single PDF byte stream."""
        text = ""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def parse_questions_from_text(self, text: str, source_file: str) -> list:
        """
        Heuristic question splitter for university/board question papers.
        Extracts question number, question text, and marks.
        """
        cleaned_text = re.sub(r'\r\n', '\n', text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        # Regex to match question starts like:
        # "1.", "1)", "Q1.", "Q.1", "Question 1", "1 (a)", "11. a)"
        q_pattern = re.compile(
            r'(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*)?(\d+[\.\)]|\b\d+\s*\([a-z]\)|\b\d+[a-z][\.\)])\s*(.*?)(?=(?:\n\s*(?:Q(?:uestion)?\.?\s*)?(?:\d+[\.\)]|\b\d+\s*\([a-z]\)|\b\d+[a-z][\.\)]))|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        marks_pattern = re.compile(r'[\(\[]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\)\]]', re.IGNORECASE)

        questions = []
        matches = list(q_pattern.finditer(cleaned_text))
        
        # If standard numbered pattern fails, fallback to line splitting
        if not matches:
            lines = [l.strip() for l in cleaned_text.split('\n') if len(l.strip()) > 20 and '?' in l]
            for idx, line in enumerate(lines, 1):
                questions.append({
                    "question_num": str(idx),
                    "question_text": line,
                    "marks": 5.0,
                    "source_file": source_file
                })
            return questions

        for m in matches:
            q_num = m.group(1).strip()
            q_body = m.group(2).strip()

            if len(q_body) < 10:
                continue

            marks = None
            marks_match = marks_pattern.search(q_body)
            if marks_match:
                marks = float(marks_match.group(1))
                # remove the marks tag from the question text for cleaner representation
                q_body = marks_pattern.sub('', q_body).strip()
            else:
                marks = 2.0 if len(q_body) < 80 else 10.0  # sensible default

            questions.append({
                "question_num": q_num,
                "question_text": q_body,
                "marks": marks,
                "source_file": source_file
            })

        return questions

    def process_all_papers(self, pdf_files: dict) -> tuple:
        """
        Converts all PDFs to text and extracts structured questions.
        Returns (raw_texts_dict, questions_df).
        """
        raw_texts = {}
        all_questions = []

        for filename, pdf_bytes in pdf_files.items():
            text = self.pdf_to_text(pdf_bytes)
            raw_texts[filename] = text
            questions = self.parse_questions_from_text(text, filename)
            all_questions.extend(questions)

        df = pd.DataFrame(all_questions)
        return raw_texts, df

    def cluster_and_predict(self, df: pd.DataFrame, similarity_threshold: float = 0.72) -> pd.DataFrame:
        """
        Uses SentenceTransformer to find semantically similar questions,
        groups them, and calculates repetition frequency & weightage score.
        """
        if df.empty or len(df) == 0:
            return pd.DataFrame()

        questions = df['question_text'].tolist()
        embeddings = self.model.encode(questions, show_progress_bar=False, convert_to_numpy=True)
        
        sim_matrix = cosine_similarity(embeddings)
        n = len(questions)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster_indices = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and sim_matrix[i, j] >= similarity_threshold:
                    cluster_indices.append(j)
                    visited[j] = True
            clusters.append(cluster_indices)

        predicted_records = []
        total_papers = df['source_file'].nunique() if 'source_file' in df.columns else 1

        for c_idx, cluster in enumerate(clusters):
            sub_df = df.iloc[cluster]
            
            # Select representative question (longest or clearest)
            rep_question = max(sub_df['question_text'].tolist(), key=len)
            
            # Frequency counts
            repetition_count = len(sub_df)
            papers_appeared = sub_df['source_file'].nunique()
            avg_marks = round(sub_df['marks'].mean(), 1)
            max_marks = sub_df['marks'].max()
            
            # Probability / Weightage Score:
            # Frequency weight + Marks weight normalized by total papers
            paper_coverage_pct = round((papers_appeared / max(1, total_papers)) * 100, 1)
            # Score formula: (Repetition Count * 40%) + (Paper Coverage % * 40%) + (Marks * 20%)
            importance_score = round((repetition_count * 2.0) + (paper_coverage_pct * 0.5) + (avg_marks * 0.3), 1)

            source_papers = ", ".join(sub_df['source_file'].unique())

            predicted_records.append({
                "Cluster ID": c_idx + 1,
                "Predicted Question": rep_question,
                "Times Repeated": repetition_count,
                "Unique Papers": papers_appeared,
                "Paper Coverage (%)": paper_coverage_pct,
                "Avg Marks": avg_marks,
                "Max Marks": max_marks,
                "Importance Score": importance_score,
                "Found In Papers": source_papers,
                "Variants Count": len(cluster)
            })

        result_df = pd.DataFrame(predicted_records)
        # Sort by Importance Score & Repetition descending
        result_df = result_df.sort_values(by=["Importance Score", "Times Repeated"], ascending=[False, False]).reset_index(drop=True)
        return result_df
