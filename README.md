# Resume Builder Assistant

An AI-enabled resume assistant for university students, built as a web prototype for **TCS Technology Day 2026 (Use Case 4)**.
It builds and scores resumes, recommends roles, answers career questions in chat, and gives a placement cell a campus-wide readiness view. All student data is **synthetic sample data**.

## Features
- **Build Resume:** live editor, auto summary, stronger project bullets, ATS-friendly **PDF export**, saved versions.
- **ATS Analyzer:** scores a resume against a job description (out of 100). Upload an existing **PDF/TXT** resume and edit it in the builder.
- **Job Match:** best-fit roles via vector search (ChromaDB), a learning plan for missing skills, and a cover letter generator.
- **Career Chat:** works offline with a built-in answer engine; optionally uses a local [Ollama](https://ollama.com) model.
- **Campus Analytics:** readiness split, branch averages, skill gaps, at-risk students, and a downloadable **PDF report**.

## How the score works (rules, so it is explainable)
| Part | Points |
|---|---|
| Required skills found | 55 |
| Keyword match with the job description (TF-IDF) | 20 |
| Resume completeness (6 checks) | 15 |
| Project bullets with numbers | 10 |

70+ is placement-ready, 50-69 almost there, below 50 needs support.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
The SQLite database (`resume.db`: tables `students`, `jobs`, `resumes`) is created automatically on first run from `data_gen.py`.
To regenerate or change the sample data, edit `data_gen.py` (or `ROLES` in `core.py`), run `python data_gen.py`, and restart the app.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit screens (5 tabs) |
| `core.py` | Scoring, resume reader, job match, chat, PDF export |
| `data_gen.py` | Generates the synthetic students and job roles into SQLite |

## Stack
Python, Streamlit, Pandas, SQLite, ChromaDB, scikit-learn, pypdf, fpdf2, Ollama (optional).

## Honest limits
- Job matching compares word patterns, not true meaning.
- Knows about 45 skills and 8 roles.
- The PDF reader works best on resumes with clear headings (SKILLS, PROJECTS...).
- Scores are a transparent approximation, not a copy of any company's ATS.

Built with AI-assisted coding (Claude Code).
