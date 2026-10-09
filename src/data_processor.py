import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        # Define expected schemas
        self.expected_question_columns = [
            'question_id', 'board', 'class_level', 'subject', 'exam_year',
            'paper_id', 'question_number', 'parent_question_id', 'question_text',
            'marks', 'chapter', 'topic', 'topic_id', 'question_type', 'difficulty',
            'source_file', 'source_page', 'syllabus_version', 'label_source'
        ]
        
    def validate_and_clean_questions(self, df):
        """
        Validates the dataframe against the expected schema.
        Adds missing columns with NaN values.
        """
        # Ensure all expected columns exist
        for col in self.expected_question_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Reorder columns
        df = df[self.expected_question_columns]
        
        # Basic cleaning
        if 'marks' in df.columns:
            # Convert marks to numeric where possible
            df['marks'] = pd.to_numeric(df['marks'], errors='coerce')
            
        if 'exam_year' in df.columns:
            df['exam_year'] = pd.to_numeric(df['exam_year'], errors='coerce')
            
        return df

    def handle_missing_topics(self, df):
        """Flags questions that don't have topics assigned."""
        df['needs_review'] = df['topic'].isna() | (df['topic'] == '')
        return df
