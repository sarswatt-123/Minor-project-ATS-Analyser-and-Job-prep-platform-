# AI-powered Resume + Job Prep — MVP

This is a minimal, local MVP built with **Streamlit**. No external APIs required.

## Features
- **ATS Resume Analyzer**: Upload PDF/DOCX + optional Job Description → get ATS score, keyword coverage, section checks, and suggestions.
- **Mock Interview (Text)**: Choose a domain (Data Science / Python / SQL / Cloud), answer questions, get immediate feedback.

## Quick Start
1. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run streamlit_app.py
```
4. Open the URL shown in the terminal (usually http://localhost:8501).

## Project Structure
```
ats_mvp/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── data/
    └── questions.json
```

## Customize
- Add or edit questions in `data/questions.json`.
- Improve scoring in `compute_ats_score()` or integrate LLMs later.
- Add login, save histories, or export reports as next steps.

## Notes
- PDF text extraction works best with **text-based PDFs** (not scanned images). Export your resume from Word/Google Docs as PDF for reliable parsing.
