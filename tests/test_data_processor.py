import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

def test_validate_and_clean_questions():
    processor = DataProcessor()
    
    # Create mock data
    data = {
        'question_text': ['Q1. What is gravity?', 'Q2. Solve x + 2 = 4'],
        'marks': ['2', '3']
    }
    df = pd.DataFrame(data)
    
    processed_df = processor.validate_and_clean_questions(df)
    
    # Check that all columns are present
    for col in processor.expected_question_columns:
        assert col in processed_df.columns
        
    # Check that marks were converted to numeric
    assert pd.api.types.is_numeric_dtype(processed_df['marks'])
    assert processed_df['marks'].iloc[0] == 2
    
def test_handle_missing_topics():
    processor = DataProcessor()
    
    data = {
        'question_text': ['Q1', 'Q2'],
        'topic': ['Physics', None]
    }
    df = pd.DataFrame(data)
    
    df = processor.handle_missing_topics(df)
    
    assert 'needs_review' in df.columns
    assert df['needs_review'].iloc[0] == False
    assert df['needs_review'].iloc[1] == True
