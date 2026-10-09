# Dataset Schemas

## 1. Questions Dataset (`questions.csv`)
- `question_id`: Unique identifier for the question.
- `board`: Educational board (e.g., CBSE).
- `class_level`: Class level (e.g., Class 10).
- `subject`: Subject name.
- `exam_year`: Year of the examination.
- `paper_id`: Identifier for the source paper.
- `question_number`: Question number in the paper.
- `parent_question_id`: If a sub-part, the ID of the parent.
- `question_text`: The extracted text.
- `marks`: Allocated marks.
- `chapter`: Associated chapter.
- `topic`: Associated topic.
- `topic_id`: Unique topic ID.
- `question_type`: Type (e.g., MCQ, Short Answer).
- `difficulty`: Difficulty level.
- `source_file`: Name of the source file.
- `source_page`: Page number in the source file.
- `syllabus_version`: Applicable syllabus year/version.
- `label_source`: How the topic was labeled (manual, AI, etc.).

## 2. Papers Dataset (`papers.csv`)
- `paper_id`: Unique identifier.
- `board`: Educational board.
- `class_level`: Class level.
- `subject`: Subject name.
- `exam_year`: Year of the exam.
- `exam_type`: Type (e.g., Final, Midterm).
- `total_marks`: Total marks for the paper.
- `source_file`: Filename.
- `processing_status`: Status of extraction.
- `syllabus_version`: Applicable syllabus.

## 3. Syllabus Dataset (`syllabus.csv`)
- `board`: Educational board.
- `class_level`: Class level.
- `subject`: Subject name.
- `syllabus_version`: Version identifier.
- `chapter`: Chapter name.
- `topic`: Topic name.
- `topic_id`: Unique identifier.
- `active_status`: Active/Inactive flag.
