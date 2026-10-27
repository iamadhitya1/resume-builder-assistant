import json

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Resume Builder Assistant", page_icon="📄", layout="wide")
core.init_db()


@st.cache_data
def cohort():
    return core.cohort_table()


students = core.load_students()
jobs = core.load_jobs()

# ---------------- sidebar ----------------
st.sidebar.title("📄 Resume Builder Assistant")
st.sidebar.caption("TCS Technology Day | Use Case 4 | sample university data")
sid = st.sidebar.selectbox("Student", students["id"], format_func=lambda i: f"{students.set_index('id').loc[i,'name']} (#{i})")
if st.session_state.get("sid") != sid:
    st.session_state.sid = sid
    st.session_state.p = core.get_profile(int(sid))
    st.session_state.msgs = []
p = st.session_state.p
role = st.sidebar.selectbox("Target role", list(core.ROLES), index=list(core.ROLES).index(p["target_role"]))
job = jobs[jobs["title"] == role].iloc[0]

models = core.local_models()
use_llm = st.sidebar.toggle("Use Ollama LLM in chat", value=False, help="Off = fast offline rule engine")
model = st.sidebar.selectbox("Model", models) if (use_llm and models) else None
st.sidebar.caption("Ollama: " + (", ".join(models) if models else "not running (offline mode)"))

if st.session_state.get("loaded_upload"):
    st.sidebar.success(f"Editing uploaded resume: {st.session_state.loaded_upload}. Open the 'Build Resume' tab.")

t1, t2, t3, t4, t5 = st.tabs(["1. Build Resume", "2. ATS Analyzer", "3. Job Match", "4. Career Chat", "5. Campus Analytics"])

# ---------------- 1. builder ----------------
with t1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Profile")
        p["name"] = st.text_input("Name", p["name"])
        a, b, c = st.columns(3)
        p["branch"] = a.text_input("Branch", p["branch"])
        p["year"] = b.number_input("Year", 1, 4, int(p["year"]))
        p["cgpa"] = c.number_input("CGPA / CPI", 0.0, 10.0, float(p["cgpa"]))
        p["links"] = st.text_input("GitHub / LinkedIn", p.get("links", ""))
        if st.button("✨ Auto-generate summary"):
            p["summary"] = core.generate_summary(p, role)
        p["summary"] = st.text_area("Summary", p.get("summary", ""), height=90)
        p["skills"] = [s.strip() for s in st.text_input("Skills (comma separated)", ", ".join(p["skills"])).split(",") if s.strip()]
        proj = st.text_area("Projects (one bullet per line)", "\n".join(p["projects"]), height=120)
        p["projects"] = [x for x in proj.split("\n") if x.strip()]
        if st.button("🚀 Improve project bullets (action verbs + metric prompts)"):
            p["projects"] = [core.improve_bullet(x) for x in p["projects"]]
            st.rerun()
        certs = st.text_area("Certifications (one per line)", "\n".join(p["certs"]), height=70)
        p["certs"] = [x for x in certs.split("\n") if x.strip()]
    with c2:
        st.subheader("Live resume preview")
        r = core.ats_score(p, job["description"], job["skills"])
        st.metric(f"ATS score for {role}", f"{r['total']}/100", core.readiness(r["total"]))
        st.components.v1.html(core.render_resume_html(p), height=520, scrolling=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Download PDF", core.render_resume_pdf(p), f"{p['name'].replace(' ', '_')}_resume.pdf",
                           "application/pdf", type="primary")
        d2.download_button("⬇ HTML", core.render_resume_html(p), f"{p['name'].replace(' ', '_')}_resume.html", "text/html")
        if d3.button("💾 Save version"):
            core.save_resume_version(int(sid), r["total"], role, core.resume_text(p))
            st.toast("Saved to SQLite")
        h = core.resume_history(int(sid))
        if len(h):
            st.caption("Version history (score improves as you edit)")
            st.line_chart(h.set_index("id")["score"])

# ---------------- 2. ATS ----------------
with t2:
    st.subheader("Resume vs job description")
    jd = st.text_area("Paste a job description (or use the sample for the selected role)", job["description"], height=110)
    up = st.file_uploader("Or upload an existing resume (PDF / TXT) to score it instead", type=["pdf", "txt"])
    target = p
    if up:
        try:
            target = core.profile_from_text(core.read_uploaded(up))
            st.info(f"Scoring uploaded file '{up.name}': {len(target['skills'])} skills detected")
            if st.button("✏ Edit this resume in the builder", type="primary"):
                new = core.builder_profile(target, p)
                p.clear()
                p.update(new)
                st.session_state.loaded_upload = up.name
                st.rerun()
        except Exception as e:
            st.warning(f"Could not read file ({e}); showing the built resume instead.")
    r = core.ats_score(target, jd, None if jd != job["description"] else job["skills"])
    m = st.columns(5)
    m[0].metric("Overall", f"{r['total']}/100")
    m[1].metric("Skills match", f"{r['skill_pct']:.0%}")
    m[2].metric("Keyword match", f"{r['keyword_pct']:.0%}")
    m[3].metric("Sections", f"{r['section_pct']:.0%}")
    m[4].metric("Impact bullets", f"{r['impact_pct']:.0%}")
    st.progress(min(1.0, r["total"] / 100))
    a, b = st.columns(2)
    a.success("Matched: " + (", ".join(r["matched"]) or "none"))
    b.error("Missing: " + (", ".join(r["missing"]) or "none"))
    st.markdown("**Recommendations**")
    for t in r["tips"] or ["Resume looks strong for this role."]:
        st.write("•", t)
    st.markdown("**Section checklist**")
    st.write(" ".join(("✅ " if ok else "❌ ") + k + "  " for k, ok in r["checks"].items()))

# ---------------- 3. job match ----------------
with t3:
    st.subheader("Best-fit roles (vector search over job descriptions)")
    rec = core.recommend_jobs(p, jobs, 8)
    st.caption(f"Retrieval engine: {rec['engine'].iloc[0]}  |  Fit = 70% skill overlap + 30% semantic similarity")
    st.bar_chart(rec.set_index("Role")["Fit %"])
    st.dataframe(rec.drop(columns="engine"), width="stretch", hide_index=True)
    st.subheader(f"Learning plan for {role}")
    r = core.ats_score(p, job["description"], job["skills"])
    for s in r["missing"] or []:
        st.write(f"**{s}** → {core.LEARN_TASKS.get(s, 'Build a mini project with it and add a resume bullet.')}")
    if not r["missing"]:
        st.write("No skill gaps. Focus on quantifying impact.")
    st.subheader("Cover letter")
    if st.button("✉ Generate cover letter for this role"):
        st.session_state.cl = core.cover_letter(p, role, r["matched"], r["missing"])
    if st.session_state.get("cl"):
        st.text_area("Cover letter (editable)", st.session_state.cl, height=220)
        st.download_button("⬇ Download cover letter", st.session_state.cl, "cover_letter.txt")

# ---------------- 4. chat ----------------
with t4:
    st.subheader("Career assistant")
    qs = ["What is my ATS score?", "What skills am I missing?", "Which jobs fit me?", "Rewrite my project bullets", "Interview questions?"]
    cols = st.columns(len(qs))
    clicked = next((q for col, q in zip(cols, qs) if col.button(q, key=q)), None)
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about your resume, skills, jobs...") or clicked
    if q:
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.spinner("Thinking..."):
            ans, src = core.chat_reply(q, p, role, jobs, model)
        st.session_state.msgs.append({"role": "assistant", "content": ans + f"\n\n*source: {src}*"})
        st.rerun()

# ---------------- 5. analytics ----------------
with t5:
    st.subheader("Campus placement readiness (placement-cell view)")
    df = cohort()
    k = st.columns(4)
    k[0].metric("Students", len(df))
    k[1].metric("Avg ATS score", f"{df['ATS score'].mean():.1f}")
    k[2].metric("Placement-ready", f"{(df['Status']=='Placement-ready').mean():.0%}")
    k[3].metric("Need support", int((df["Status"] == "Needs support").sum()))
    a, b = st.columns(2)
    a.markdown("**Readiness by status**")
    a.bar_chart(df["Status"].value_counts())
    b.markdown("**Average ATS score by branch**")
    b.bar_chart(df.groupby("Branch")["ATS score"].mean())
    # skill gap: market demand vs cohort supply
    demand = pd.Series([s for sk in jobs["skills"] for s in sk]).value_counts()
    have = pd.Series([s for _, s_ in students["profile_json"].items() for s in json.loads(s_)["skills"]]).value_counts()
    gap = pd.DataFrame({"Demand (roles asking)": demand, "Students with skill": have}).fillna(0).loc[demand.index[:12]]
    gap["Students with skill"] = (gap["Students with skill"] / len(students) * 100).round(0)
    st.markdown("**Top in-demand skills vs % of students who have them**")
    st.dataframe(gap, width="stretch")
    miss = pd.Series([s.strip() for x in df["Missing skills"] for s in x.split(",") if s.strip()]).value_counts().head(10)
    st.markdown("**Most common skill gaps across campus (training priorities)**")
    st.bar_chart(miss)
    st.markdown("**At-risk students (report)**")
    risk = df[df["Status"] == "Needs support"].sort_values("ATS score")
    st.dataframe(risk, width="stretch", hide_index=True)
    r1, r2 = st.columns(2)
    r1.download_button("⬇ Download campus report (PDF)", core.render_campus_pdf(df), "campus_readiness_report.pdf",
                       "application/pdf", type="primary")
    r2.download_button("⬇ Download at-risk list (CSV)", risk.to_csv(index=False), "at_risk_students.csv", "text/csv")
