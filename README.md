---
title: ExamPredict AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# ExamPredict AI — AI-Based Question Paper Analysis and Prediction System

ExamPredict AI is an educational dashboard designed to analyze historical school examination papers, identify recurring topics, and help students prioritize their revision.

## Features

- **Upload and Parse:** Extract questions from past PDF question papers.
- **Topic Analysis:** Map questions to syllabus topics and visualize historical trends.
- **Predictive Ranking:** Use simple baseline heuristics and optional Machine Learning to rank topics by likelihood of appearance.
- **Practice Paper Generator:** Automatically assemble a practice paper from historical questions based on a specific blueprint.
- **Important Questions:** Review the most highly recommended topics and past questions.

## Limitations

- **No Crystal Ball:** The system cannot predict the exact questions that will appear on a future exam.
- **Data Dependent:** Meaningful ML predictions require a substantial dataset of labeled historical papers.
- **Extraction Artifacts:** PDF text extraction is not always perfect, especially with complex formatting like equations or multi-column layouts.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application:**
   ```bash
   python app.py
   ```
3. **Run Tests:**
   ```bash
   pytest tests/
   ```

## Dataset Format

The system expects structured CSVs or cleanly parsed PDFs.
See `data/README.md` for schema details.
