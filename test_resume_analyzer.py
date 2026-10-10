"""Regression tests for the resume scorer.

Run:  python test_resume_analyzer.py
Whenever you find a resume that gets a wrong score, paste it into RESUMES with the
score range you believe is right.  That is how the scorer stays honest over time.
"""
import sys

try:
    from resume_analyzer import analyze_resume
except ImportError:  # allow running from another folder
    sys.path.insert(0, ".")
    from resume_analyzer import analyze_resume

STRONG_SWE = """Priya Sharma
priya.sharma@gmail.com | +91 98765 43210 | linkedin.com/in/priyasharma | github.com/priyas

SUMMARY
Backend engineer with 4 years of experience building payment and data platforms.

EXPERIENCE
Senior Software Engineer | FinServe Technologies | Jan 2022 - Present
• Architected an event-driven payments pipeline using Kafka and PostgreSQL that processes 2.5M transactions per day with 99.99% uptime.
• Reduced API p95 latency from 800ms to 120ms by introducing Redis caching and query optimisation.
• Led a team of 5 engineers to migrate 40 microservices from EC2 to Kubernetes, cutting infrastructure cost by 32%.
• Built CI/CD pipelines with GitHub Actions and Terraform, decreasing release time from 2 days to 45 minutes.
• Mentored 3 junior developers, two of whom were promoted within 12 months.

Software Engineer | DataWorks | Jul 2019 - Dec 2021
• Developed a fraud-detection service in Python and scikit-learn that flagged 18% more fraudulent transactions.
• Implemented REST APIs in FastAPI consumed by 12 internal teams.
• Automated monthly reconciliation reports, saving 60 analyst hours per month.

EDUCATION
B.Tech in Computer Science, IIT Hyderabad, 2015 - 2019, CGPA 8.7/10

SKILLS
Languages: Python, Java, SQL, Go
Tools: Kafka, Kubernetes, Docker, Terraform, AWS, PostgreSQL, Redis, FastAPI

CERTIFICATIONS
AWS Certified Solutions Architect - Associate
"""

PROSE_STRONG = """Arjun Rao
arjun.rao@example.com  +91 90000 11111  github.com/arjunrao

Professional Experience
Machine Learning Engineer, Visionary Labs (2021 - Present)
I designed and deployed a computer-vision defect detection system for a manufacturing client, which cut manual inspection time by 70% and was rolled out across 6 factories. Alongside this I built the training pipeline in PyTorch and MLflow, retraining models weekly on 200,000 labelled images. I also mentored two interns and presented results to the executive team each quarter.

Data Scientist, Retail Insights (2018 - 2021)
Developed a demand forecasting model in Python that improved forecast accuracy by 14% across 300 stores. Built dashboards in Tableau used daily by 40 category managers, and automated data cleaning that saved 15 hours per week.

Education
M.Sc. Statistics, University of Pune, 2018

Skills
Python, PyTorch, scikit-learn, SQL, MLflow, Tableau, Docker, AWS
"""

FRESHER_GOOD = """Ananya Reddy
ananya.reddy@gmail.com | 9876501234 | github.com/ananyar | linkedin.com/in/ananyareddy

EDUCATION
B.Tech Computer Science, JNTU Hyderabad, 2020 - 2024, CGPA 8.4

PROJECTS
Movie Recommendation System
- Built a hybrid recommender using collaborative filtering and TF-IDF content features on the MovieLens 1M dataset, achieving RMSE of 0.87.
- Deployed the model as a Flask API on Heroku and created a React front end used by 50 classmates.

Real-time Chat Application
- Developed a chat app with Node.js, Socket.IO and MongoDB supporting 200 concurrent users.
- Implemented JWT authentication and role-based access control.

INTERNSHIP
Web Development Intern, TechNova, Jun 2023 - Aug 2023
- Created 6 responsive pages in React that improved mobile Lighthouse score from 62 to 91.
- Fixed 25 UI bugs and wrote unit tests with Jest.

SKILLS
Python, JavaScript, React, Node.js, MongoDB, SQL, Git, Flask

ACHIEVEMENTS
- Won 2nd place among 120 teams at the college hackathon 2023
"""

WEAK_GENERIC = """John Doe
john@mail.com

Objective
Hardworking team player looking for a good job in a dynamic company where I can use my skills.

Experience
ABC Company
- Responsible for testing software
- Worked on various projects
- Helped the team with many tasks
- Handled customer issues
- Participated in meetings

Education
BSc, 2015

Skills
MS Office, communication, teamwork, hardworking
"""

KEYWORD_STUFFED = """Rahul Verma
rahul.v@mail.com | 9988776655

EXPERIENCE
Company X, Jan 2020 - Present
- Spearheaded synergy infrastructure strategy governance optimization.
- Orchestrated championed streamlined architected cross-functional enterprise-wide lifecycle.
- Pioneered overhauled conceptualized scalable end-to-end solutions.
- Optimized streamlined accelerated delivered results.

EDUCATION
B.Com, 2018

SKILLS
Leadership, Strategy, Governance, Synergy
"""

MESSY_PDF = """SNEHA KULKARNI
sneha.k@outlook.com    +91-99887-76655

WORK EXPERIENCE
DATA ANALYST    Mar 2021 – Present
Fintrack Solutions, Pune
\uf0b7 Led a cross-functional effort to build automated Power BI dashboards that gave
leadership real-time visibility into 15 KPIs and reduced manual reporting by 20 hours
per week.
\uf0b7 Analysed 3 years of customer churn data with SQL and Python, identifying drivers
that helped retention team lower churn by 9%.
\uf0b7 Built an ETL workflow in Airflow loading 1.2 million records nightly into Snowflake.
\uf0b7 Presented findings to senior management and trained 25 analysts on self-serve BI.

EDUCATION
MBA (Analytics), Symbiosis Pune, 2019 - 2021

TECHNICAL SKILLS
SQL, Python, Power BI, Airflow, Snowflake, Excel, Tableau
"""

MARKETING_GOOD = """Meera Nair
meera.nair@gmail.com | +91 91234 56780 | linkedin.com/in/meeranair

Experience
Digital Marketing Manager - BrightBrand (Apr 2020 - Present)
- Grew organic traffic by 145% in 12 months through an SEO content strategy and 80 pillar articles.
- Managed an annual paid media budget of Rs 1.2 crore, lowering cost per lead by 28%.
- Led a team of 6 content creators and designers across 3 brands.
- Launched an email nurture program that lifted conversion from 2.1% to 4.6%.
- Partnered with sales to build lead scoring that increased qualified leads by 35%.

Marketing Executive - Zenith Media (Jun 2017 - Mar 2020)
- Planned and executed 24 social campaigns reaching 3 million users.
- Produced weekly analytics reports for 10 clients.

Education
MBA Marketing, NMIMS Mumbai, 2017

Skills
SEO, Google Analytics, HubSpot, Google Ads, Meta Ads, Copywriting, A/B testing
"""

MARKETING_WEAK = """Ravi Kumar
ravi@mail.com 9876543210

Experience
XYZ Pvt Ltd
- Responsible for social media
- Helped with marketing campaigns
- Worked on content
- Assisted manager in various tasks

Education
BBA 2016

Skills
Social media, MS Office
"""

TINY = "Developer. Python. Looking for work."

# name: (text, min_score, max_score)
RESUMES = {
    "strong software engineer (bullets)": (STRONG_SWE, 80, 99),
    "strong ML engineer (prose, no bullets)": (PROSE_STRONG, 68, 99),
    "good fresher (projects, few metrics)": (FRESHER_GOOD, 68, 92),
    "messy PDF export, good content": (MESSY_PDF, 70, 99),
    "strong marketing (non-tech)": (MARKETING_GOOD, 78, 99),
    "weak generic resume": (WEAK_GENERIC, 0, 42),
    "weak marketing resume": (MARKETING_WEAK, 0, 42),
    "keyword-stuffed nonsense": (KEYWORD_STUFFED, 0, 42),
    "tiny / almost empty": (TINY, 0, 30),
}


def run():
    failures = 0
    print(f"{'resume':45s} {'score':>5s}  expected   breakdown (struct/depth/action/impact)")
    for name, (text, lo, hi) in RESUMES.items():
        r = analyze_resume(text)
        b = r["breakdown"]
        ok = lo <= r["score"] <= hi
        failures += not ok
        print(f"{name:45s} {r['score']:5d}  {lo:>3d}-{hi:<3d} {'OK ' if ok else 'FAIL'} "
              f"{b['structure']}/{b['depth']}/{b['action']}/{b['impact']}  bullets={r['bullet_count']}")
    # ordering sanity: good must beat bad
    s = {k: analyze_resume(v[0])["score"] for k, v in RESUMES.items()}
    pairs = [("strong software engineer (bullets)", "weak generic resume"),
             ("strong marketing (non-tech)", "weak marketing resume"),
             ("messy PDF export, good content", "keyword-stuffed nonsense"),
             ("good fresher (projects, few metrics)", "weak generic resume")]
    for good, bad in pairs:
        ok = s[good] > s[bad] + 15
        failures += not ok
        print(f"order check: {good} > {bad} + 15 ... {'OK' if ok else 'FAIL'}")
    print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
