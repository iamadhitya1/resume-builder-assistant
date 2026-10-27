"""Core logic for the Resume Builder Assistant: data, ATS scoring, matching, chat."""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = Path(__file__).parent / "resume.db"

SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "sql", "react", "node.js", "html", "css",
    "django", "flask", "fastapi", "spring boot", "git", "docker", "kubernetes", "aws", "azure",
    "linux", "rest api", "mongodb", "postgresql", "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "power bi", "tableau", "excel",
    "statistics", "data structures", "algorithms", "ci/cd", "selenium", "testing", "agile",
    "communication", "embedded c", "iot", "langchain", "streamlit",
]

# Market demand tag used in the dashboard (share of sample jobs that ask for it)
WEAK_VERBS = {
    "worked on": "Built", "helped": "Contributed to", "responsible for": "Owned",
    "made": "Developed", "did": "Executed", "used": "Applied", "handled": "Managed",
}

LEARN_TASKS = {
    "docker": "Containerise one of your projects with a Dockerfile (about 2 hours) and list it as a bullet.",
    "kubernetes": "Deploy a Docker image to a local minikube cluster and write 3 lines about it.",
    "aws": "Host a static site or a small API on the AWS free tier and note the services used.",
    "azure": "Deploy a small web app on Azure for Students and note the services used.",
    "sql": "Solve 20 SQL joins/aggregation problems and add a project that uses a real database.",
    "git": "Push all projects to GitHub with README, commits and a clear folder structure.",
    "ci/cd": "Add a GitHub Actions workflow that runs tests on every push.",
    "react": "Build a small dashboard in React that calls one of your APIs.",
    "machine learning": "Train a model on a public dataset; report accuracy/F1 in a bullet.",
    "deep learning": "Fine-tune a small CNN/transformer and report the metric improvement.",
    "nlp": "Build a text-classifier or RAG chatbot and report retrieval accuracy.",
    "rest api": "Build a CRUD API with auth and document it with Swagger/OpenAPI.",
    "testing": "Add unit tests to a project and report coverage percentage.",
    "statistics": "Write up a hypothesis-test analysis in a notebook on a real dataset.",
    "power bi": "Build a 2-page dashboard on a public dataset and publish a screenshot.",
    "tableau": "Build a public Tableau dashboard and link it from the resume.",
    "data structures": "Solve 50 problems on arrays, trees and graphs; mention the platform and count.",
    "algorithms": "Solve 50 problems on DP and graphs; mention the platform and count.",
    "langchain": "Build a document Q&A bot with LangChain and a vector store.",
}

ROLES = {
    "Software Developer": ["java", "python", "data structures", "algorithms", "sql", "git", "rest api", "agile"],
    "Full Stack Developer": ["javascript", "react", "node.js", "html", "css", "mongodb", "rest api", "git"],
    "Data Analyst": ["python", "sql", "pandas", "excel", "power bi", "statistics", "communication"],
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "statistics", "sql"],
    "AI/ML Engineer": ["python", "deep learning", "pytorch", "nlp", "machine learning", "docker", "git"],
    "Cloud/DevOps Engineer": ["linux", "docker", "kubernetes", "aws", "ci/cd", "git", "python"],
    "QA/Test Engineer": ["testing", "selenium", "java", "sql", "agile", "git", "ci/cd"],
    "Embedded/IoT Engineer": ["embedded c", "c++", "iot", "linux", "python", "git"],
}

JOB_DESCRIPTIONS = {
    "Software Developer": "Design, code and test scalable backend services. Strong in data structures, algorithms and OOP. Work in agile teams with git and REST APIs on SQL databases.",
    "Full Stack Developer": "Build responsive web apps using React, Node.js and MongoDB. Create REST APIs, write clean HTML/CSS/JavaScript and collaborate through git.",
    "Data Analyst": "Turn raw data into insights using Python, SQL, pandas and Excel. Build Power BI dashboards, apply statistics and present findings with strong communication.",
    "Data Scientist": "Build predictive models with machine learning in Python using pandas, numpy and scikit-learn. Apply statistics, run experiments and query data with SQL.",
    "AI/ML Engineer": "Develop and deploy deep learning and NLP models with PyTorch. Productionise machine learning pipelines using Docker and git.",
    "Cloud/DevOps Engineer": "Automate infrastructure and deployments on AWS using Docker, Kubernetes and CI/CD pipelines on Linux. Script in Python and manage code in git.",
    "QA/Test Engineer": "Design test plans, automate tests using Selenium and Java, validate data with SQL, run CI/CD pipelines and work in agile sprints.",
    "Embedded/IoT Engineer": "Develop firmware in embedded C and C++, integrate IoT sensors and gateways, work on Linux and script tooling in Python.",
}


# ---------- database ----------
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db(force=False):
    if DB_PATH.exists() and not force:
        return
    from data_gen import build
    build()


def load_students() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql("SELECT * FROM students", c)
    return df


def get_profile(student_id: int) -> dict:
    with conn() as c:
        row = c.execute("SELECT profile_json FROM students WHERE id=?", (student_id,)).fetchone()
    return json.loads(row["profile_json"])


def load_jobs() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql("SELECT * FROM jobs", c)
    df["skills"] = df["skills_json"].apply(json.loads)
    return df


def save_resume_version(student_id: int, score: float, role: str, text: str):
    with conn() as c:
        c.execute(
            "INSERT INTO resumes(student_id, role, score, text, created) VALUES(?,?,?,?,datetime('now','localtime'))",
            (student_id, role, score, text),
        )


def resume_history(student_id: int) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql(
            "SELECT id, role, score, created FROM resumes WHERE student_id=? ORDER BY id", c, params=(student_id,)
        )


# ---------- text and skills ----------
def extract_skills(text: str) -> list[str]:
    t = text.lower()
    found = []
    for s in SKILLS:
        if re.search(r"(?<![a-z0-9+#.])" + re.escape(s) + r"(?![a-z0-9+#])", t):
            found.append(s)
    return found


def resume_text(p: dict) -> str:
    parts = [p.get("name", ""), p.get("summary", ""), f"{p.get('branch','')} CGPA/CPI {p.get('cgpa','')}",
             "Skills: " + ", ".join(p.get("skills", []))]
    parts += ["Project: " + x for x in p.get("projects", [])]
    parts += ["Certification: " + x for x in p.get("certs", [])]
    parts.append(p.get("links", ""))
    return "\n".join(x for x in parts if x)


def improve_bullet(b: str) -> str:
    out = b.strip()
    low = out.lower()
    for weak, strong in WEAK_VERBS.items():
        if low.startswith(weak):
            out = strong + out[len(weak):]
            break
    if not re.search(r"\d", out) and "[add a metric" not in out:
        out += "  [add a metric: users, % gain, speed-up, dataset size]"
    return out


def generate_summary(p: dict, role: str) -> str:
    top = ", ".join(p.get("skills", [])[:4]) or "core programming skills"
    return (f"{p.get('branch','')} student (CGPA/CPI {p.get('cgpa','')}) targeting a {role} role, "
            f"with hands-on project experience in {top}. Delivers working, measurable projects and "
            f"learns new tools fast.")


# ---------- ATS scoring ----------
def ats_score(p: dict, jd_text: str, required: list[str] | None = None) -> dict:
    text = resume_text(p)
    req = required or extract_skills(jd_text)
    have = set(extract_skills(text)) | set(s.lower() for s in p.get("skills", []))
    matched = [s for s in req if s in have]
    missing = [s for s in req if s not in have]
    skill_pct = len(matched) / max(1, len(req))

    try:
        m = TfidfVectorizer(stop_words="english").fit_transform([text, jd_text])
        sim = float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        sim = 0.0
    keyword_pct = min(1.0, sim * 2.5)

    checks = {
        "Professional summary": len(p.get("summary", "")) > 40,
        "5+ skills listed": len(p.get("skills", [])) >= 5,
        "2+ projects": len(p.get("projects", [])) >= 2,
        "Certification": len(p.get("certs", [])) >= 1,
        "GitHub / LinkedIn link": bool(re.search(r"github|linkedin", p.get("links", ""), re.I)),
        "CGPA/CPI mentioned": bool(p.get("cgpa")),
    }
    section_pct = sum(checks.values()) / len(checks)

    projects = p.get("projects", [])
    impact_pct = (sum(1 for x in projects if re.search(r"\d", x)) / len(projects)) if projects else 0.0

    total = 55 * skill_pct + 20 * keyword_pct + 15 * section_pct + 10 * impact_pct
    tips = []
    if missing:
        tips.append(f"Add or learn these missing skills: {', '.join(missing)}.")
    if impact_pct < 0.7:
        tips.append("Quantify project bullets (numbers, %, users, dataset size).")
    for k, ok in checks.items():
        if not ok:
            tips.append(f"Missing section/element: {k}.")
    weak = [x for x in projects if any(x.lower().startswith(w) for w in WEAK_VERBS)]
    if weak:
        tips.append("Replace weak openers ('worked on', 'helped') with strong action verbs.")
    return {
        "total": round(total, 1), "skill_pct": skill_pct, "keyword_pct": keyword_pct,
        "section_pct": section_pct, "impact_pct": impact_pct, "matched": matched,
        "missing": missing, "checks": checks, "tips": tips,
    }


def read_uploaded(file) -> str:
    """Extract text from an uploaded PDF or text resume."""
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(file).pages)
    return file.read().decode("utf-8", errors="ignore")


def profile_from_text(text: str) -> dict:
    p = _parse_text(text)
    em = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    br = re.search(r"B\.?\s?(?:Tech|E)\.?[ ,]+(?:in\s+)?([A-Za-z&. ]+?)\s*(?:,|\||\n|Year|\(|$)", text, re.I)
    yr = re.search(r"Year\s*([1-4])", text, re.I)
    p.update({"email": em.group(0) if em else "", "branch": br.group(1).strip() if br else "",
              "year": yr.group(1) if yr else ""})
    return p


def _parse_text(text: str) -> dict:
    """Rough parse of free-text resume into a profile dict so the same scorer can run on it."""
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l.strip(" -•\t") for l in raw]
    cg = re.search(r"(?:cgpa|cpi)[^0-9]*([0-9]\.?[0-9]*)", text, re.I)
    bullet = re.compile(r"^[-•*▪–]\s*")
    sections, cur = {}, None
    for l in raw:
        if l.isupper() and len(l) < 30:
            cur = l.lower()
            sections[cur] = []
        elif cur:
            has_marker = bool(bullet.match(l))
            clean_l = bullet.sub("", l).strip()
            # a wrapped line (no bullet) in a bulleted section continues the previous bullet
            if sections[cur] and not has_marker and cur in ("projects", "certifications") \
                    and any(bullet.match(x) for x in raw if x != l):
                sections[cur][-1] += " " + clean_l
            else:
                sections[cur].append(clean_l)
    sec = lambda key: next((v for k, v in sections.items() if key in k), None)
    if sec("project") is not None:
        return {
            "name": lines[0], "summary": " ".join(sec("summary") or sec("objective") or [])[:300],
            "skills": extract_skills(text), "cgpa": cg.group(1) if cg else "", "branch": "", "year": "",
            "projects": sec("project")[:6], "certs": (sec("cert") or [])[:4],
            "links": " ".join(re.findall(r"(?:github|linkedin)\.com/\S+", text, re.I)),
        }
    return {
        "name": lines[0] if lines else "Uploaded resume", "summary": " ".join(lines[1:3])[:300],
        "skills": extract_skills(text), "cgpa": cg.group(1) if cg else "", "branch": "", "year": "",
        "projects": [l for l in lines if len(l) > 40 and not re.search(r"certif|course", l, re.I)][:6],
        "certs": [l for l in lines if re.search(r"certif|nptel|coursera|udemy", l, re.I)][:4],
        "links": " ".join(re.findall(r"(?:github|linkedin)\.com/\S+", text, re.I)),
    }


def cover_letter(p: dict, role: str, matched: list[str], missing: list[str]) -> str:
    proj = (p.get("projects") or ["my academic projects"])[0].split("  [add")[0]
    skills = ", ".join(matched[:4]) or ", ".join(p.get("skills", [])[:4])
    learn = f" I am also actively upskilling in {', '.join(missing[:2])}." if missing else ""
    return (f"Dear Hiring Manager,\n\nI am a {p.get('branch','')} student (CGPA/CPI {p.get('cgpa','')}) applying for the "
            f"{role} position. My strengths in {skills} match your requirements. For example: {proj}.{learn}\n\n"
            f"I would welcome the chance to contribute and grow with your team.\n\nSincerely,\n{p.get('name','')}")


def builder_profile(parsed: dict, base: dict) -> dict:
    """Make a parsed upload safe to load into the builder form (fills in defaults)."""
    try:
        cg = float(parsed.get("cgpa") or 0)
    except ValueError:
        cg = 0.0
    out = dict(base)
    out.update(parsed)
    out.update({
        "cgpa": cg if 0 <= cg <= 10 else 0.0,
        "year": int(parsed["year"]) if str(parsed.get("year", "")).isdigit() and 1 <= int(parsed["year"]) <= 4 else 3,
        "branch": parsed.get("branch") or base.get("branch", ""),
        "email": parsed.get("email", ""),
        "links": parsed.get("links", ""),
    })
    return out


def readiness(score: float) -> str:
    return "Placement-ready" if score >= 70 else "Almost there" if score >= 50 else "Needs support"


# ---------- job recommendation (ChromaDB vector store, offline hashing embeddings) ----------
_HV = HashingVectorizer(n_features=1024, ngram_range=(1, 2), alternate_sign=False, norm="l2", stop_words="english")
_COLL = None


def _collection(jobs: pd.DataFrame):
    global _COLL
    if _COLL is None:
        import chromadb
        client = chromadb.EphemeralClient()
        coll = client.get_or_create_collection("jobs", metadata={"hnsw:space": "cosine"})
        docs = (jobs["title"] + ". " + jobs["description"]).tolist()
        coll.upsert(ids=[str(i) for i in jobs["id"]], documents=docs,
                    embeddings=_HV.transform(docs).toarray().tolist())
        _COLL = coll
    return _COLL


def recommend_jobs(p: dict, jobs: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    text = resume_text(p)
    have = set(extract_skills(text)) | set(s.lower() for s in p.get("skills", []))
    try:
        res = _collection(jobs).query(query_embeddings=_HV.transform([text]).toarray().tolist(),
                                      n_results=len(jobs))
        sem = {int(i): 1 - d for i, d in zip(res["ids"][0], res["distances"][0])}
        engine = "ChromaDB"
    except Exception:
        docs = (jobs["title"] + ". " + jobs["description"]).tolist()
        sims = cosine_similarity(_HV.transform([text]), _HV.transform(docs))[0]
        sem = dict(zip(jobs["id"], sims))
        engine = "cosine"
    rows = []
    for _, j in jobs.iterrows():
        req = j["skills"]
        matched = [s for s in req if s in have]
        overlap = len(matched) / len(req)
        score = 100 * (0.7 * overlap + 0.3 * min(1.0, sem[int(j["id"])] * 3))
        rows.append({"Role": j["title"], "Fit %": round(score, 1), "Skills matched": f"{len(matched)}/{len(req)}",
                     "Missing": ", ".join(s for s in req if s not in have) or "none", "engine": engine})
    return pd.DataFrame(rows).sort_values("Fit %", ascending=False).head(top_k).reset_index(drop=True)


# ---------- cohort analytics ----------
def cohort_table() -> pd.DataFrame:
    students, jobs = load_students(), load_jobs()
    by_title = {r["title"]: r for _, r in jobs.iterrows()}
    rows = []
    for _, s in students.iterrows():
        p = json.loads(s["profile_json"])
        job = by_title[s["target_role"]]
        r = ats_score(p, job["description"], job["skills"])
        rows.append({"id": s["id"], "Name": s["name"], "Branch": s["branch"], "Year": s["year"],
                     "CGPA": s["cgpa"], "Target role": s["target_role"], "ATS score": r["total"],
                     "Status": readiness(r["total"]), "Missing skills": ", ".join(r["missing"])})
    return pd.DataFrame(rows)


# ---------- conversational assistant ----------
def local_models() -> list[str]:
    try:
        import ollama
        return [m.model for m in ollama.Client(timeout=3).list().models]
    except Exception:
        return []


def _facts(p: dict, role: str, jobs: pd.DataFrame) -> dict:
    job = jobs[jobs["title"] == role].iloc[0]
    r = ats_score(p, job["description"], job["skills"])
    return {"role": role, "ats": r, "recs": recommend_jobs(p, jobs, 3), "job": job}


def rule_reply(q: str, p: dict, role: str, jobs: pd.DataFrame) -> str:
    f = _facts(p, role, jobs)
    r, ql = f["ats"], q.lower()
    if re.search(r"score|ats|rate|how good", ql):
        return (f"Your resume scores **{r['total']}/100** for **{role}** ({readiness(r['total'])}). "
                f"Skills match {r['skill_pct']:.0%}, keywords {r['keyword_pct']:.0%}, "
                f"sections {r['section_pct']:.0%}, impact bullets {r['impact_pct']:.0%}.\n\n"
                + "\n".join(f"- {t}" for t in r["tips"][:4]))
    if re.search(r"gap|missing|learn|skill|improve", ql):
        if not r["missing"]:
            return "No skill gaps for this role. Focus on quantifying your project bullets."
        return "Skill gaps and a 1-week fix plan:\n" + "\n".join(
            f"- **{s}**: {LEARN_TASKS.get(s, 'Build a mini project using it and add one resume bullet.')}"
            for s in r["missing"][:5])
    if re.search(r"job|role|recommend|suit|fit", ql):
        return "Best-fit roles for you:\n" + "\n".join(
            f"- **{x['Role']}**: {x['Fit %']}% fit (missing: {x['Missing']})" for _, x in f["recs"].iterrows())
    if re.search(r"summary|objective|about", ql):
        return "Suggested summary:\n\n> " + generate_summary(p, role)
    if re.search(r"project|bullet|rewrite", ql):
        return "Stronger project bullets:\n" + "\n".join(f"- {improve_bullet(b)}" for b in p.get("projects", [])[:3])
    if re.search(r"interview|prepare|question", ql):
        qs = [f"Explain how you used **{s}** in your projects." for s in r["matched"][:3]]
        return "Likely interview questions from your own resume:\n" + "\n".join(f"- {x}" for x in qs) + \
               "\n- Walk me through your best project end to end."
    return ("I can help with: **ATS score**, **skill gaps**, **job recommendations**, **summary**, "
            "**project bullets** and **interview questions**. Try one.")


def chat_reply(q: str, p: dict, role: str, jobs: pd.DataFrame, model: str | None) -> tuple[str, str]:
    """Return (answer, source). Uses an Ollama model grounded on computed facts; falls back to rules."""
    base = rule_reply(q, p, role, jobs)
    if not model:
        return base, "rule-based"
    try:
        import ollama
        prompt = (f"You are a career coach for a university student. Target role: {role}.\n"
                  f"Resume:\n{resume_text(p)}\n\nVerified analysis (do not contradict):\n{base}\n\n"
                  f"Student question: {q}\nAnswer in under 150 words, concrete and encouraging.")
        out = ollama.Client(timeout=25).chat(model=model, messages=[{"role": "user", "content": prompt}])
        return out["message"]["content"], f"LLM ({model})"
    except Exception:
        return base, "rule-based (LLM unavailable)"


def render_resume_pdf(p: dict) -> bytes:
    """Clean single-column, ATS-friendly PDF (real text, no tables/images)."""
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 14, 18)
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    fonts = Path("C:/Windows/Fonts")
    if (fonts / "arial.ttf").exists():
        pdf.add_font("R", "", str(fonts / "arial.ttf"))
        pdf.add_font("R", "B", str(fonts / "arialbd.ttf"))
        fam, clean = "R", (lambda s: str(s))
    else:
        fam, clean = "Helvetica", (lambda s: str(s).encode("latin-1", "replace").decode("latin-1"))
    ink, accent = (25, 25, 25), (31, 78, 121)
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(t):
        pdf.ln(3)
        pdf.set_font(fam, "B", 11)
        pdf.set_text_color(*accent)
        pdf.cell(W, 6, t, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*accent)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
        pdf.ln(1.5)
        pdf.set_text_color(*ink)

    def para(t, size=10):
        pdf.set_font(fam, "", size)
        pdf.multi_cell(W, 5, clean(t), new_x="LMARGIN", new_y="NEXT")

    def bullets(items):
        pdf.set_font(fam, "", 10)
        for it in items:
            x = pdf.l_margin
            pdf.set_x(x)
            pdf.cell(5, 5, "\u2022" if fam == "R" else "-")
            pdf.multi_cell(W - 5, 5, clean(it), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.8)

    pdf.set_text_color(*ink)
    pdf.set_font(fam, "B", 22)
    pdf.cell(W, 10, clean(p.get("name", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fam, "", 9.5)
    pdf.set_text_color(90, 90, 90)
    contact = " | ".join(x for x in [p.get("email", ""), p.get("links", "")] if x)
    pdf.cell(W, 5, clean(contact), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*ink)
    if p.get("summary"):
        heading("SUMMARY")
        para(p["summary"])
    heading("EDUCATION")
    edu = f"B.Tech {p.get('branch','')}".strip()
    if p.get("year"):
        edu += f", Year {p['year']}"
    if p.get("cgpa"):
        edu += f"  |  CGPA/CPI: {p['cgpa']}"
    para(edu)
    heading("SKILLS")
    para(", ".join(p.get("skills", [])))
    if p.get("projects"):
        heading("PROJECTS")
        bullets([x.split("  [add a metric")[0] for x in p["projects"]])
    if p.get("certs"):
        heading("CERTIFICATIONS")
        bullets(p["certs"])
    return bytes(pdf.output())


def render_campus_pdf(df: pd.DataFrame) -> bytes:
    """One-page-style placement-cell report: KPIs, branch view, top skill gaps, at-risk students."""
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_margins(15, 14, 15)
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    fonts = Path("C:/Windows/Fonts")
    fam = "Helvetica"
    if (fonts / "arial.ttf").exists():
        pdf.add_font("R", "", str(fonts / "arial.ttf"))
        pdf.add_font("R", "B", str(fonts / "arialbd.ttf"))
        fam = "R"
    accent = (31, 78, 121)
    W = pdf.w - 30

    def h(t):
        pdf.ln(3)
        pdf.set_font(fam, "B", 12)
        pdf.set_text_color(*accent)
        pdf.cell(W, 7, t, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(25, 25, 25)

    def row(cells, widths, bold=False, fill=False):
        pdf.set_font(fam, "B" if bold else "", 9)
        if fill:
            pdf.set_fill_color(225, 235, 245)
        for c, w in zip(cells, widths):
            pdf.cell(w, 6, str(c)[: int(w / 1.9)], border=1, fill=fill)
        pdf.ln()

    pdf.set_font(fam, "B", 20)
    pdf.cell(W, 10, "Campus Placement Readiness Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fam, "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(W, 5, "Resume Builder Assistant | sample university data | generated " +
             pd.Timestamp.now().strftime("%d %b %Y %H:%M"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(25, 25, 25)

    h("Summary")
    n = len(df)
    ready = (df["Status"] == "Placement-ready").sum()
    almost = (df["Status"] == "Almost there").sum()
    need = (df["Status"] == "Needs support").sum()
    pdf.set_font(fam, "", 10)
    pdf.multi_cell(W, 6, f"{n} students assessed. Average ATS score {df['ATS score'].mean():.1f}/100. "
                         f"Placement-ready: {ready} ({ready/n:.0%}); almost there: {almost}; needs support: {need}.",
                   new_x="LMARGIN", new_y="NEXT")

    h("Average score by branch")
    for b, v in df.groupby("Branch")["ATS score"].mean().sort_values(ascending=False).items():
        pdf.set_font(fam, "", 10)
        pdf.cell(40, 6, b)
        pdf.set_fill_color(*accent)
        pdf.cell(max(1, v * 1.0), 4, "", fill=True)
        pdf.cell(20, 6, f"  {v:.1f}", new_x="LMARGIN", new_y="NEXT")

    h("Top skill gaps across campus (teach these first)")
    miss = pd.Series([s.strip() for x in df["Missing skills"] for s in x.split(",") if s.strip()]).value_counts().head(8)
    for s, c in miss.items():
        pdf.set_font(fam, "", 10)
        pdf.cell(W, 6, f"- {s}: missing in {c} students ({c/n:.0%})", new_x="LMARGIN", new_y="NEXT")

    h("Recommended actions")
    pdf.set_font(fam, "", 10)
    top3 = ", ".join(miss.index[:3]) if len(miss) else "n/a"
    pdf.multi_cell(W, 6, f"1) Run short workshops on: {top3}.  2) Hold a resume clinic for the {need} students "
                         f"flagged 'Needs support'.  3) Re-run this report after training to measure the score lift.",
                   new_x="LMARGIN", new_y="NEXT")

    h(f"At-risk students (lowest 15 of {need})")
    widths = [10, 36, 20, 46, 16, 52]
    row(["ID", "Name", "Branch", "Target role", "Score", "Top missing"], widths, bold=True, fill=True)
    for _, r in df[df["Status"] == "Needs support"].sort_values("ATS score").head(15).iterrows():
        row([r["id"], r["Name"], r["Branch"], r["Target role"], r["ATS score"],
             ", ".join(r["Missing skills"].split(", ")[:2])], widths)
    return bytes(pdf.output())


def render_resume_html(p: dict) -> str:
    li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
    return f"""<html><head><meta charset="utf-8"><title>{p['name']} - Resume</title>
<style>html{{background:#ffffff}}body{{font-family:Calibri,Arial,sans-serif;max-width:780px;margin:0 auto;padding:24px 32px;background:#ffffff;color:#111111}}
p,li,div{{color:#111111}}.m{{color:#444444!important}}
h1{{margin:0}}h2{{border-bottom:2px solid #1f4e79;color:#1f4e79;font-size:15px;margin-top:18px}}
.m{{color:#555;font-size:13px}}li{{margin:3px 0;font-size:13px}}</style></head><body>
<h1>{p['name']}</h1><div class="m">{p.get('email','')} | {p.get('links','')}</div>
<h2>SUMMARY</h2><p style="font-size:13px">{p.get('summary','')}</p>
<h2>EDUCATION</h2><p style="font-size:13px">B.Tech {p.get('branch','')}, Year {p.get('year','')} | CGPA/CPI {p.get('cgpa','')}</p>
<h2>SKILLS</h2><p style="font-size:13px">{', '.join(p.get('skills', []))}</p>
<h2>PROJECTS</h2><ul>{li(p.get('projects', []))}</ul>
<h2>CERTIFICATIONS</h2><ul>{li(p.get('certs', []))}</ul></body></html>"""
