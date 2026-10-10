"""Independent sanity benchmark: hand-written resumes in natural phrasing (not sampled
from domain_keywords.json). Compares old vs new matcher. Usage: python eval_realistic.py [old_dir]"""
import sys, importlib.util, random, numpy as np
from pathlib import Path
import ml_domain_matcher as new

BE = {"Backend & Microservices Engineering", "Software Engineering & Backend"}
CASES = [
("Rahul Verma\nBackend Engineer\nBuilt REST APIs with Django and FastAPI serving 2M requests a day. Designed PostgreSQL schemas, added Redis caching, and moved the monolith to microservices on Docker and AWS. Implemented JWT authentication, Celery workers and RabbitMQ queues. Wrote unit testing suites and maintained CI/CD pipelines with GitHub Actions. Mentored two junior developers in code reviews.", BE),
("Ananya Rao\nMachine Learning Engineer\nTrained gradient boosting and random forest models with scikit-learn and XGBoost for churn prediction. Did feature engineering in pandas and numpy, ran hyperparameter tuning, and deployed models behind a Flask service. Performed EDA, A/B testing and model monitoring. Python, SQL, Jupyter. MSc in Statistics.", {"Data Science & Machine Learning","Data Science & Statistical Modeling"}),
("Tom Becker\nFrontend Developer\nBuilt responsive single page applications in React and TypeScript with Redux and Tailwind CSS. Optimised webpack bundles, wrote Jest and React Testing Library tests, and improved Lighthouse accessibility scores. Collaborated with designers in Figma and shipped features weekly in an agile team.", {"Frontend & Web Engineering","Frontend & Single Page Apps"}),
("Karthik S\nDevOps Engineer\nAutomated infrastructure with Terraform and Ansible on AWS. Ran Kubernetes clusters with Helm, set up Jenkins and GitLab CI pipelines, and built Prometheus and Grafana dashboards. Reduced deployment time by 70 percent. On-call rotation, Linux administration, shell scripting.", {"Cloud & DevOps Engineering","DevOps & CI/CD Engineering","Site Reliability Engineering (SRE)"}),
("Mary Thomas RN\nStaff Nurse, Medical-Surgical Ward\nProvided bedside nursing care for 12 patients per shift: vital signs monitoring, medication administration, IV cannulation, wound dressing and catheterization. Wrote nursing care plans, completed patient handover using SBAR, and educated families on discharge planning. BLS certified.", {"Clinical Nursing & Acute Care"}),
("Suresh Iyer, CA\nChartered Accountant\nHandled statutory audit and internal audit engagements. Prepared financial statements under Ind AS, performed bank reconciliation, general ledger review and month-end close in Tally Prime. Managed GST returns and TDS filings, accounts payable and receivable, and variance analysis for the finance team.", {"Accounting, Audit & Assurance","Corporate Taxation & Compliance"}),
("Neha Kapoor\nTechnical Recruiter\nSourced engineers on LinkedIn Recruiter and GitHub, ran screening calls, managed the applicant tracking system, coordinated interview loops and negotiated offers. Closed 40 technical hires a year, built talent pipelines and employer branding campaigns with hiring managers.", {"Technical Talent Acquisition & Sourcing","Human Resources & Talent Acquisition"}),
("Imran Khan\nSite Engineer - Civil\nSupervised RCC construction on a 12 storey building. Structural analysis and design in STAAD Pro and ETABS, prepared BOQ and quantity takeoff, reviewed drawings in AutoCAD, tracked schedules in Primavera P6 and ensured IS 456 compliance. Quality control and safety on site.", {"Civil & Structural Engineering"}),
("Divya Menon\nAndroid Developer\nDeveloped Android apps in Kotlin using Jetpack Compose, MVVM, Coroutines, Room and Retrofit. Dependency injection with Hilt, Firebase push notifications, Play Store releases, unit tests with JUnit and Espresso. Reduced crash rate with Crashlytics.", {"Android Mobile Development"}),
("Chris Wong\nSOC Analyst\nMonitored alerts in Splunk SIEM, triaged phishing and malware incidents, performed threat hunting mapped to MITRE ATT&CK and wrote incident response playbooks. Used CrowdStrike EDR, Wireshark and Sentinel for investigation and log analysis. Escalated and documented incidents.", {"SOC Analysis & Incident Response","Cybersecurity & Information Security","Cybersecurity & Ethical Hacking"}),
("Pooja Nair\nDigital Marketing Manager\nPlanned SEO and paid campaigns on Google Ads and Meta Ads, improved conversion rate optimization on landing pages, ran email marketing automation and tracked KPIs in GA4. Grew organic traffic 150 percent and cut CAC by 30 percent. Content calendar and social media strategy.", {"Digital Marketing & Growth"}),
("Aisha Rahman\nProduct Manager\nOwned the product roadmap, wrote user stories and PRDs, prioritised the backlog with RICE, and ran A/B tests with analytics. Worked with engineering and design in agile sprints, tracked OKRs in Jira and aligned stakeholders. Launched a B2B onboarding flow that lifted activation by 18 percent.", {"Product Management","Product Management (Technical & B2B)"}),
("Luis Ortega\nData Engineer\nBuilt batch and streaming ETL pipelines with Apache Airflow, Spark and Kafka, modelled warehouse tables in Snowflake with dbt, and optimised SQL. Managed data lake storage on S3 in Parquet, added data quality checks and orchestrated daily loads.", {"Data Engineering & Pipelines","Data Engineering & Big Data","Big Data & Distributed Analytics"}),
("Sara Ali\nBI Developer\nCreated Power BI dashboards with DAX and Power Query on top of SQL Server. Built star schema data models, executive KPI reports and row level security. Automated monthly MIS reporting from Excel and presented insights to stakeholders.", {"Business Intelligence & Reporting"}),
("Helen Brooks\nPrimary School Teacher\nPlanned and delivered engaging lessons in literacy and numeracy for year 3. Classroom management, phonics, differentiated instruction, assessment and progress tracking, and regular parent-teacher meetings. Led the school reading club and supported children with additional needs.", {"Primary & Elementary Education","Special Education & Instructional Specialist"}),
("Vikram Desai\nMechanical Design Engineer\nDesigned components in SolidWorks and CATIA, applied GD&T on drawings, ran FEA simulations in ANSYS, and supported CNC machining with DFM reviews. Reduced part cost by 12 percent. Prototyping, tolerance stack up, bill of materials and ECN process.", {"Mechanical Design & Manufacturing (CAD/CAM)","Mechanical Engineering"}),
("Julia Novak\nQA Automation Engineer\nBuilt Selenium and Cypress UI test frameworks in Java, API tests with Rest Assured and Postman, integrated suites into Jenkins CI. Wrote test plans, tracked defects in Jira, performed regression testing and reduced release cycle time with test automation.", {"QA Automation & SDET","Manual QA & Usability Testing"}),
("Mei Tanaka\nUX Designer\nConducted user research and usability testing, created personas, journey maps, wireframes and high fidelity prototypes in Figma. Maintained the design system, ran accessibility reviews and worked with developers on interaction design for a mobile banking app.", {"UI/UX & Interaction Design","User Research & Usability"}),
("Arjun Pillai\nEmbedded Firmware Engineer\nWrote C firmware for STM32 and ARM Cortex-M microcontrollers with FreeRTOS. Implemented I2C, SPI, UART and CAN drivers, debugged with oscilloscope and JTAG, and bring-up of custom PCBs. Low power design and bootloader development.", {"Embedded Systems & Microcontrollers","Firmware & Hardware Integration"}),
("Adv. Rohan Mehta\nLitigation Associate\nDrafted pleadings and written submissions, conducted legal research, reviewed contracts, appeared before the High Court and prepared case strategy. Managed discovery, client counselling and settlement negotiations in commercial disputes.", {"Legal Advisory & Litigation"}),
("Emily Carter\nInvestment Banking Analyst\nBuilt DCF, LBO and trading comps models in Excel, prepared pitch books and CIMs, supported M&A transactions through due diligence and closing. Valuation, financial modeling, and client presentations for sell-side mandates.", {"Investment Banking & M&A Advisory","Investment Banking & Corporate Finance","Financial Modeling & Valuation"}),
("Dr. Anil Gupta\nConsultant Radiologist\nReported CT, MRI, ultrasound and X-ray studies, performed image-guided biopsies, supervised radiographers, and contributed to tumour board meetings. Experienced in PACS, contrast protocols and radiation safety. Teaching residents and clinical audits.", {"Radiology & Diagnostic Imaging"}),
("Kenji Sato\nSolidity Developer\nWrote and audited Solidity smart contracts on Ethereum, built dApps with ethers.js and Hardhat, implemented ERC20 and ERC721 tokens, optimised gas, tested with Foundry and integrated MetaMask wallets. Security reviews with Slither.", {"Blockchain & Smart Contracts"}),
("Lena Fischer\nSupply Chain Planner\nDemand planning and forecasting in SAP, managed procurement and supplier negotiations, inventory optimisation, S&OP meetings and logistics coordination. Reduced stock-outs by 20 percent and lowered purchasing costs through vendor consolidation.", {"Supply Chain Planning & Procurement","Warehouse Logistics & Freight Forwarding"}),
]
# non-resume prose: must NOT be classified confidently
NONRESUME = "Preheat the oven to 180 degrees and grease a round cake tin. Whisk the eggs and sugar until pale, fold in the flour gently, then bake for thirty five minutes until golden. Leave to cool on a rack before adding cream and fresh strawberries, and serve with a hot cup of tea in the afternoon."

def load_old(d):
    sys.path.insert(0, d); spec = importlib.util.spec_from_file_location("old_m", Path(d)/"ml_domain_matcher.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    old = load_old(sys.argv[1]) if len(sys.argv) > 1 else None
    names = new.get_domain_names(); eng = new.get_engine(); rng = random.Random(0)
    n_ok = o_ok = n_conf = 0; fails = []
    for txt, ok in CASES:
        r = new.detect_best_domain(txt); n_ok += r["best_domain"] in ok; n_conf += r["confident"]
        if r["best_domain"] not in ok: fails.append((txt.split("\n")[1], r["best_domain"], r["ranked_domains"][:3]))
        if old:
            o_ok += old.detect_best_domain(txt)["best_domain"] in ok
    n = len(CASES)
    print(f"Detection (accepting twin domains): NEW {n_ok}/{n}" + (f"   OLD {o_ok}/{n}" if old else "") + f"   (NEW confident on {n_conf}/{n})")
    for f in fails: print("  NEW miss:", f)
    # match-score tiers
    tiers = {"same":[], "related":[], "unrelated":[]}; otiers = {"same":[], "related":[], "unrelated":[]}
    for txt, ok in CASES:
        true = sorted(ok, key=lambda d: -new.analyze(txt)["prob"][names.index(d)])[0]
        ti = names.index(true)
        rel = [names[j] for j in range(len(names)) if j != ti and eng.rel[ti, j] >= 0.4]
        unrel = [names[j] for j in range(len(names)) if eng.rel[ti, j] < 0.05]
        picks = {"same": [true], "related": rel[:3], "unrelated": rng.sample(unrel, 4)}
        for t, ds in picks.items():
            for d in ds:
                tiers[t].append(new.match_score(txt, d)["score"])
                if old: otiers[t].append(old.screen_for_domain(txt, d)["skills"]["coverage"])
    print("\nMatch score by tier (mean | min..max)   preferred role = true / closely related / unrelated")
    for t in tiers:
        a = np.array(tiers[t]); line = f"  NEW {t:10} {a.mean():5.1f} | {a.min()}..{a.max()}"
        if old: b = np.array(otiers[t]); line += f"      OLD {b.mean():5.1f} | {b.min()}..{b.max()}"
        print(line)
    def auc(pos, neg): return np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    print(f"  NEW AUC same-vs-unrelated {auc(tiers['same'],tiers['unrelated']):.3f}   related-vs-unrelated {auc(tiers['related'],tiers['unrelated']):.3f}   same-vs-related {auc(tiers['same'],tiers['related']):.3f}")
    if old: print(f"  OLD AUC same-vs-unrelated {auc(otiers['same'],otiers['unrelated']):.3f}   related-vs-unrelated {auc(otiers['related'],otiers['unrelated']):.3f}   same-vs-related {auc(otiers['same'],otiers['related']):.3f}")
    r = new.detect_best_domain(NONRESUME); print("\nNon-resume prose ->", r["best_domain"], "| confident:", r["confident"])
    if old: o = old.detect_best_domain(NONRESUME); print("   old ->", o["best_domain"], o["best_score"])
if __name__ == "__main__": main()
