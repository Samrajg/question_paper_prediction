import pandas as pd

class TopicAnalyzer:
    def __init__(self):
        pass

    def get_topic_frequency(self, questions_df, subject_filter=None, year_filter=None):
        """
        Calculates topic frequency by exam year.
        """
        df = questions_df.copy()
        
        # Apply filters
        if subject_filter and subject_filter != "All":
            df = df[df['subject'] == subject_filter]
            
        if year_filter and year_filter != "All":
            df = df[df['exam_year'] == int(year_filter)]
            
        if df.empty or 'topic' not in df.columns or df['topic'].dropna().empty:
            return pd.DataFrame()
            
        # Group by topic and calculate frequency
        freq = df.groupby(['topic', 'exam_year']).size().reset_index(name='count')
        return freq

    def get_marks_distribution(self, questions_df, subject_filter=None):
        """
        Calculates marks distribution by topic.
        """
        df = questions_df.copy()
        if subject_filter and subject_filter != "All":
            df = df[df['subject'] == subject_filter]
            
        if df.empty or 'topic' not in df.columns or 'marks' not in df.columns or df['topic'].dropna().empty:
            return pd.DataFrame()
            
        marks_dist = df.groupby('topic')['marks'].sum().reset_index(name='total_marks')
        marks_dist = marks_dist.sort_values('total_marks', ascending=False)
        return marks_dist

    def get_baseline_ranking(self, questions_df):
        """
        Ranks topics based on simple historical recurrence and recency.
        """
        if questions_df.empty or 'topic' not in questions_df.columns:
            return pd.DataFrame(columns=["Rank", "Topic", "Historical Frequency", "Recency Score"])
            
        # Frequency
        freq = questions_df['topic'].value_counts().reset_index()
        freq.columns = ['Topic', 'Historical Frequency']
        
        # Recency (average year)
        recency = questions_df.groupby('topic')['exam_year'].mean().reset_index()
        recency.columns = ['Topic', 'Avg Year']
        
        # Merge
        ranking = pd.merge(freq, recency, on='Topic')
        
        # Simple score: Normalize frequency and multiply by a recency factor
        # For simplicity, just sort by frequency for baseline
        ranking = ranking.sort_values('Historical Frequency', ascending=False).reset_index(drop=True)
        ranking['Rank'] = ranking.index + 1
        ranking['Recency Score'] = ranking['Avg Year']
        
        return ranking[['Rank', 'Topic', 'Historical Frequency', 'Recency Score']]
