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

## Download and run it again (step by step)

You need **Python 3.10 or newer** (built on Python 3.14) and **git** (or just a browser to download a ZIP).

**1. Get the code** (pick one)

- With git (PowerShell, Command Prompt or any terminal):
  ```bash
  git clone https://github.com/iamadhitya1/resume-builder-assistant.git
  cd resume-builder-assistant
  ```
- Without git: open the repo page, click the green **Code** button, then **Download ZIP**, unzip it, and open a terminal inside the unzipped folder.

**2. Install the packages** (a virtual environment keeps your other Python projects clean)

- Windows PowerShell:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- macOS / Linux:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

**3. Start the app**
```bash
python -m streamlit run app.py
```
Open http://localhost:8501 and wait about 10 seconds on the first load. Press `Ctrl+C` in the terminal to stop it.

**4. Try the demo**
- Open the **ATS Analyzer** tab, choose the role **Data Analyst**, and upload `samples/Rohan_Verma_WEAK_resume.pdf` (about 35/100).
- Click **Edit this resume in the builder**, fix it in the **Build Resume** tab, then **Download PDF**.
- Upload `samples/Rohan_Verma_IMPROVED_resume.pdf` to see about 96/100.

**Optional AI chat:** install [Ollama](https://ollama.com), pull a model (for example `ollama pull llama3`), then switch on "Use Ollama LLM in chat" in the sidebar. Everything else works without it.

**Next time you open it:** activate the environment again (step 2, the activate line only) and run step 3.

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

## License
All rights reserved by M. Adhitya. Anyone can read the code and run the app as it is, but it may not be copied, modified, redistributed or claimed as someone else's work without permission. The libraries it depends on keep their own licenses (listed in LICENSE). See [LICENSE](LICENSE).
