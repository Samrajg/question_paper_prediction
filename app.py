import gradio as gr
import pandas as pd
import numpy as np
import os
import plotly.express as px

from src.pdf_extractor import PDFExtractor
from src.data_processor import DataProcessor
from src.topic_analyzer import TopicAnalyzer

# Create dummy directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Global instances
extractor = PDFExtractor()
processor = DataProcessor()
analyzer = TopicAnalyzer()

def process_uploaded_files(board, class_level, subject, year, pdf_files, csv_file, current_data):
    if current_data is None:
        current_data = pd.DataFrame(columns=processor.expected_question_columns)
        
    new_data = pd.DataFrame()
    
    if pdf_files:
        for pdf in pdf_files:
            filename = os.path.basename(pdf.name)
            df = extractor.process_pdf(pdf.name, filename)
            df['board'] = board
            df['class_level'] = class_level
            df['subject'] = subject
            df['exam_year'] = int(year) if year else np.nan
            new_data = pd.concat([new_data, df], ignore_index=True)
            
    if csv_file:
        try:
            df = pd.read_csv(csv_file.name)
            df['board'] = board
            df['class_level'] = class_level
            df['subject'] = subject
            if 'exam_year' not in df.columns:
                df['exam_year'] = int(year) if year else np.nan
            new_data = pd.concat([new_data, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
    if not new_data.empty:
        new_data = processor.validate_and_clean_questions(new_data)
        updated_data = pd.concat([current_data, new_data], ignore_index=True)
    else:
        updated_data = current_data
        
    # Generate dashboard stats
    num_papers = updated_data['source_file'].nunique() if 'source_file' in updated_data.columns else 0
    num_questions = len(updated_data)
    num_topics = updated_data['topic'].nunique() if 'topic' in updated_data.columns else 0
    years = updated_data['exam_year'].dropna().unique() if 'exam_year' in updated_data.columns else []
    years_str = ", ".join([str(int(y)) for y in sorted(years)]) if len(years) > 0 else "None"
    status = "Processed successfully" if num_questions > 0 else "No valid questions found"
    
    return updated_data, num_papers, num_questions, num_topics, years_str, status, updated_data

def update_topic_analysis(df, subject, year):
    if df is None or df.empty:
        return None, None, pd.DataFrame()
        
    freq_df = analyzer.get_topic_frequency(df, subject, year)
    if not freq_df.empty:
        fig_freq = px.bar(freq_df, x='topic', y='count', color='exam_year', title="Topic Frequency")
    else:
        fig_freq = None
        
    marks_df = analyzer.get_marks_distribution(df, subject)
    if not marks_df.empty:
        fig_marks = px.pie(marks_df, values='total_marks', names='topic', title="Marks Distribution")
    else:
        fig_marks = None
        
    filtered_df = df.copy()
    if subject and subject != "All":
        filtered_df = filtered_df[filtered_df['subject'] == subject]
    if year and year != "All":
        filtered_df = filtered_df[filtered_df['exam_year'] == int(year)]
        
    display_cols = ['topic', 'question_text', 'exam_year', 'marks']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    return fig_freq, fig_marks, filtered_df[available_cols].head(50)

def calculate_baseline(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return analyzer.get_baseline_ranking(df)

def create_dashboard(state_df):
    with gr.Tab("Dashboard"):
        gr.Markdown("## ExamPredict AI Dashboard")
        gr.Markdown("Understand past papers. Prioritise revision. Practise smarter.")
        
        with gr.Row():
            imported_papers_count = gr.Number(value=0, label="Imported Papers", interactive=False)
            extracted_questions_count = gr.Number(value=0, label="Extracted Questions", interactive=False)
            distinct_topics_count = gr.Number(value=0, label="Distinct Syllabus Topics", interactive=False)
        
        with gr.Row():
            available_years = gr.Textbox(value="None", label="Available Exam Years", interactive=False)
            processing_status = gr.Textbox(value="Idle", label="Data Processing Status", interactive=False)
        
        gr.Markdown("""
        ### What this system CAN and CANNOT predict
        **CAN:** This system analyses historical exam papers to identify frequently tested topics, chapter weightings, and historical trends. It can rank topics based on past recurrence.
        **CANNOT:** This system CANNOT predict exact questions that will appear in an upcoming examination. It is a revision prioritization tool, not a crystal ball.
        """)
        
        return imported_papers_count, extracted_questions_count, distinct_topics_count, available_years, processing_status
        
def create_upload_manage(state_df, d_stats):
    with gr.Tab("Upload and Manage Papers"):
        gr.Markdown("## Upload Data")
        with gr.Row():
            with gr.Column():
                board_dropdown = gr.Dropdown(choices=["CBSE", "ICSE", "State Board", "Other"], label="Board")
                class_dropdown = gr.Dropdown(choices=["Class 10", "Class 12"], label="Class")
                subject_dropdown = gr.Dropdown(choices=["Mathematics", "Science", "Physics", "Chemistry", "Biology"], label="Subject")
                year_dropdown = gr.Dropdown(choices=[str(y) for y in range(2010, 2026)], label="Exam Year")
                
                pdf_upload = gr.File(label="Upload PDF Question Paper(s)", file_types=[".pdf"], file_count="multiple")
                csv_upload = gr.File(label="Upload Structured Questions CSV", file_types=[".csv"])
                
                upload_btn = gr.Button("Process Files", variant="primary")
                
            with gr.Column():
                gr.Markdown("### Extracted Questions")
                questions_table = gr.Dataframe(interactive=True)
                download_csv_btn = gr.Button("Download as CSV")

        upload_btn.click(
            fn=process_uploaded_files,
            inputs=[board_dropdown, class_dropdown, subject_dropdown, year_dropdown, pdf_upload, csv_upload, state_df],
            outputs=[state_df, d_stats[0], d_stats[1], d_stats[2], d_stats[3], d_stats[4], questions_table]
        )
        return questions_table

def create_topic_analysis(state_df):
    with gr.Tab("Topic Analysis"):
        gr.Markdown("## Topic Analysis and Trends")
        gr.Markdown("Historical statistics based on processed papers.")
        
        with gr.Row():
            filter_subject = gr.Dropdown(choices=["All", "Mathematics", "Science", "Physics", "Chemistry", "Biology"], label="Subject Filter", value="All")
            filter_year = gr.Dropdown(choices=["All"] + [str(y) for y in range(2010, 2026)], label="Year Filter", value="All")
            analyze_btn = gr.Button("Update Analysis")
            
        with gr.Row():
            topic_freq_plot = gr.Plot(label="Topic Frequency by Year")
            marks_dist_plot = gr.Plot(label="Marks Distribution by Topic")
            
        gr.Markdown("### Questions Associated with Topics")
        topic_questions_table = gr.Dataframe()

        analyze_btn.click(
            fn=update_topic_analysis,
            inputs=[state_df, filter_subject, filter_year],
            outputs=[topic_freq_plot, marks_dist_plot, topic_questions_table]
        )

def create_ml_prediction(state_df):
    with gr.Tab("ML Prediction"):
        gr.Markdown("## Topic Ranking and Prediction")
        
        with gr.Accordion("Stage 1: Historical Baseline", open=True):
            gr.Markdown("Ranks topics based on simple historical recurrence and recency.")
            baseline_btn = gr.Button("Calculate Baseline Ranking")
            baseline_table = gr.Dataframe(headers=["Rank", "Topic", "Historical Frequency", "Recency Score"])
            
            baseline_btn.click(
                fn=calculate_baseline,
                inputs=[state_df],
                outputs=[baseline_table]
            )
            
        with gr.Accordion("Stage 2: Supervised ML Prediction", open=False):
            gr.Markdown("Trains a classifier (Logistic Regression / Random Forest) to predict topic appearance based on historical data.")
            gr.Markdown("*Note: Requires sufficient labelled historical examples. Chronological splitting is used.*")
            
            model_type = gr.Radio(["Logistic Regression", "Random Forest"], value="Logistic Regression", label="Model Selection")
            train_btn = gr.Button("Train & Evaluate Model")
            ml_results_text = gr.Textbox(label="Evaluation Report", lines=5)
            ml_predictions_table = gr.Dataframe(headers=["Topic", "Predicted Appearance Probability", "Expected Marks"])
            
            def mock_train(df):
                if df is None or len(df) < 10:
                    return "Insufficient data for ML training. Please upload more labeled historical data.", pd.DataFrame()
                return "Training successful (mocked).", pd.DataFrame([{"Topic": "Sample", "Probability": 0.8, "Marks": 5}])
                
            train_btn.click(
                fn=mock_train,
                inputs=[state_df],
                outputs=[ml_results_text, ml_predictions_table]
            )

def create_important_questions():
    with gr.Tab("Important Questions"):
        gr.Markdown("## High Priority Questions & Topics")
        gr.Markdown("Explore the highest ranked topics and their historical questions.")
        
        priority_table = gr.Dataframe(
            headers=["Topic/Chapter", "Historical Frequency", "Years Appeared", "Historical Marks", "Reason for Ranking", "Source Reference", "Type"]
        )

def create_practice_paper():
    with gr.Tab("Practice Paper Generator"):
        gr.Markdown("## Generate Practice Paper")
        
        with gr.Row():
            with gr.Column():
                board_sel = gr.Dropdown(choices=["CBSE", "ICSE"], label="Board")
                class_sel = gr.Dropdown(choices=["Class 10", "Class 12"], label="Class")
                subject_sel = gr.Dropdown(choices=["Mathematics", "Science"], label="Subject")
                total_marks = gr.Number(value=80, label="Total Marks")
                num_questions = gr.Number(value=20, label="Number of Questions")
                
                generate_btn = gr.Button("Generate Practice Paper", variant="primary")
            
            with gr.Column():
                generated_paper_display = gr.Markdown("### Your Practice Paper will appear here")
                download_pdf_btn = gr.Button("Download as PDF (Printable)")
                download_docx_btn = gr.Button("Download as DOCX")
                
        def mock_generate(marks, num_q):
            return f"**Practice Paper**\nTotal Marks: {marks}\nQuestions: {num_q}\n\n*This is a placeholder for the generated paper based on historical questions.*"
            
        generate_btn.click(
            fn=mock_generate,
            inputs=[total_marks, num_questions],
            outputs=[generated_paper_display]
        )

def create_evaluation_reports():
    with gr.Tab("Evaluation and Model Reports"):
        gr.Markdown("## Evaluation Metrics")
        
        eval_metrics = gr.Textbox(
            label="Model Performance Metrics", 
            value="Run ML Training in the 'ML Prediction' tab to see metrics.",
            lines=10
        )
        
        gr.Markdown("""
        ### Limitations
        * **Dataset Size:** ML models require significant historical data. Small datasets lead to unreliable predictions.
        * **Syllabus Changes:** Historical data may not perfectly align with the current syllabus.
        * **Extraction Quality:** OCR and PDF extraction errors can introduce noise into the training data.
        """)

def main():
    with gr.Blocks(title="ExamPredict AI", theme=gr.themes.Soft()) as app:
        # State variable to hold data across the app
        state_df = gr.State(None)
        
        gr.Markdown("# ExamPredict AI: AI-Based Question Paper Analysis and Prediction System")
        
        d_stats = create_dashboard(state_df)
        create_upload_manage(state_df, d_stats)
        create_topic_analysis(state_df)
        create_ml_prediction(state_df)
        create_important_questions()
        create_practice_paper()
        create_evaluation_reports()
        
    app.launch()

if __name__ == "__main__":
    main()
