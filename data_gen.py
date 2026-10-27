"""Generate synthetic sample university data (no real students) into SQLite + CSV."""
import json
import random
import sqlite3

import pandas as pd

import core

FIRST = ["Aarav", "Ananya", "Karthik", "Priya", "Rahul", "Sneha", "Vikram", "Divya", "Arjun", "Meera",
         "Rohit", "Lakshmi", "Suresh", "Pooja", "Naveen", "Kavya", "Manoj", "Nisha", "Harish", "Swathi"]
LAST = ["Kumar", "Sharma", "Reddy", "Iyer", "Nair", "Patel", "Singh", "Menon", "Das", "Rao", "Gupta", "Pillai"]
BRANCHES = ["CSE", "IT", "ECE", "AI&DS", "Mechanical"]

PROJECTS = {
    "Software Developer": ["Worked on a library management system in Java with SQL backend",
                           "Built a REST API for student records handling 500+ requests/day"],
    "Full Stack Developer": ["Built a React + Node.js expense tracker used by 120 classmates",
                             "Helped develop a portfolio website with MongoDB"],
    "Data Analyst": ["Built a Power BI dashboard on sales data covering 50,000 rows",
                     "Did analysis of student attendance using pandas"],
    "Data Scientist": ["Trained a churn model with scikit-learn reaching 87% accuracy",
                       "Worked on house price prediction"],
    "AI/ML Engineer": ["Built a PyTorch image classifier with 92% accuracy on 10 classes",
                       "Made a chatbot using NLP"],
    "Cloud/DevOps Engineer": ["Containerised a Flask app with Docker and deployed on AWS EC2",
                              "Used GitHub Actions for builds"],
    "QA/Test Engineer": ["Automated 40 test cases using Selenium and Java, cutting regression time by 60%",
                         "Handled manual testing for a college portal"],
    "Embedded/IoT Engineer": ["Built an IoT weather station with ESP32 sending data every 5 s",
                              "Worked on an Arduino line-follower robot"],
}
CERTS = ["NPTEL Python for Data Science", "AWS Cloud Practitioner", "Google Data Analytics (Coursera)",
         "Oracle Java SE Foundations", "Microsoft Azure Fundamentals", "TCS iON Career Edge"]


def build(seed: int = 7, n: int = 80):
    random.seed(seed)
    if core.DB_PATH.exists():
        core.DB_PATH.unlink()
    c = sqlite3.connect(core.DB_PATH)
    c.executescript("""
    CREATE TABLE students(id INTEGER PRIMARY KEY, name TEXT, branch TEXT, year INT, cgpa REAL,
                          target_role TEXT, profile_json TEXT);
    CREATE TABLE jobs(id INTEGER PRIMARY KEY, title TEXT, description TEXT, skills_json TEXT);
    CREATE TABLE resumes(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INT, role TEXT, score REAL,
                         text TEXT, created TEXT);
    """)
    for i, (title, skills) in enumerate(core.ROLES.items(), 1):
        c.execute("INSERT INTO jobs VALUES(?,?,?,?)",
                  (i, title, core.JOB_DESCRIPTIONS[title], json.dumps(skills)))

    rows = []
    roles = list(core.ROLES)
    for i in range(1, n + 1):
        role = random.choice(roles)
        req = core.ROLES[role]
        # strong students know most required skills, weak ones few
        ability = random.choice([0.3, 0.45, 0.6, 0.75, 0.9])
        have = [s for s in req if random.random() < ability]
        have += random.sample(core.SKILLS, k=random.randint(1, 3))
        have = list(dict.fromkeys(have))
        proj = PROJECTS[role][: random.choice([1, 2, 2])]
        name = f"{random.choice(FIRST)} {random.choice(LAST)}"
        p = {
            "name": name, "email": f"{name.split()[0].lower()}{i}@univ.edu", "branch": random.choice(BRANCHES),
            "year": random.choice([3, 4]), "cgpa": round(random.uniform(6.2, 9.4), 2), "target_role": role,
            "summary": random.choice(["", "Motivated student looking for opportunities in technology and growth."]),
            "skills": have, "projects": proj,
            "certs": random.sample(CERTS, k=random.choice([0, 1, 1, 2])),
            "links": random.choice(["", "github.com/" + name.split()[0].lower()]),
        }
        c.execute("INSERT INTO students VALUES(?,?,?,?,?,?,?)",
                  (i, name, p["branch"], p["year"], p["cgpa"], role, json.dumps(p)))
        rows.append({k: p[k] for k in ("name", "branch", "year", "cgpa", "target_role")})
    c.commit()
    c.close()
    pd.DataFrame(rows).to_csv(core.DB_PATH.with_name("sample_students.csv"), index=False)


if __name__ == "__main__":
    build()
    print("Sample data created:", core.DB_PATH)
