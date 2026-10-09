import pandas as pd
from src.topic_analyzer import TopicAnalyzer

def test_get_topic_frequency():
    analyzer = TopicAnalyzer()
    
    data = {
        'topic': ['Algebra', 'Algebra', 'Geometry', 'Calculus', 'Algebra'],
        'exam_year': [2021, 2021, 2021, 2022, 2022],
        'subject': ['Math', 'Math', 'Math', 'Math', 'Math']
    }
    df = pd.DataFrame(data)
    
    freq_df = analyzer.get_topic_frequency(df)
    
    assert not freq_df.empty
    
    # Algebra in 2021 should be 2
    algebra_2021 = freq_df[(freq_df['topic'] == 'Algebra') & (freq_df['exam_year'] == 2021)]
    assert algebra_2021['count'].iloc[0] == 2

def test_get_baseline_ranking():
    analyzer = TopicAnalyzer()
    
    data = {
        'topic': ['Algebra', 'Algebra', 'Geometry', 'Calculus', 'Algebra'],
        'exam_year': [2021, 2021, 2021, 2022, 2022]
    }
    df = pd.DataFrame(data)
    
    ranking = analyzer.get_baseline_ranking(df)
    
    assert not ranking.empty
    assert ranking['Topic'].iloc[0] == 'Algebra'
    assert ranking['Historical Frequency'].iloc[0] == 3
