"""JobShield Domain Matcher v3 - weighted, calibrated role matching.

Public API is unchanged (detect_best_domain, screen_for_domain, get_domain_names,
is_valid_resume_content) plus match_score() / analyze().

How scoring works
-----------------
1. Skill extraction is case-, spelling- and synonym-insensitive ("NodeJS", "node.js",
   "Node JS" are one skill; "OOP" == "Object Oriented Programming"; "DSA" == "Data
   Structures and Algorithms"). Variants are merged so a skill is never counted twice or
   reported as missing when the resume has it under another spelling.
2. Every skill has an IMPORTANCE inside a domain:
        importance = distinctiveness x position
   distinctiveness : a skill found in many domains (python, git, excel) is worth little,
                     a skill specific to the domain (ibnr, solidity, tavr) is worth a lot.
   position        : domain_keywords.json lists the fundamentals first, so earlier = more
                     essential.
3. Coverage is measured against what a *solid* candidate would show (the weight of the 14
   most important core skills and 8 most important supporting skills), not against the
   whole list. Core skills carry 75% of the competency score, supporting skills 25%.
4. Anti-inflation: generic-only overlap is capped, scores follow a concave curve, nothing
   exceeds 97, and a job title alone can never produce a match.
5. Domain alignment (used for detection and the ranked suggestions) = competency, discounted
   when the ML classifier believes another domain fits better, plus a small job-title bonus.
6. Match score for a preferred role = 55% core skills + 15% supporting skills +
   30% role alignment (does the whole profile - and the job title - read as that role?).
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DOMAIN_FILE = BASE_DIR / "domain_keywords.json"
MODEL_FILE = BASE_DIR / "domain_model.joblib"

# --------------------------------------------------------------------------- #
# Acronym / synonym knowledge layer
# --------------------------------------------------------------------------- #
RAW_ACRONYM_MAP = {
    "eeg": ["electroencephalogram", "electroencephalography"],
    "emg": ["electromyogram", "electromyography"],
    "ecg": ["ekg", "electrocardiogram", "electrocardiography"],
    "ncs": ["nerve conduction study", "nerve conduction studies"],
    "pcr": ["polymerase chain reaction"],
    "mri": ["magnetic resonance imaging"],
    "cpr": ["cardiopulmonary resuscitation"],
    "bls": ["basic life support"],
    "acls": ["advanced cardiovascular life support", "advanced cardiac life support"],
    "pals": ["pediatric advanced life support"],
    "nicu": ["neonatal intensive care unit"],
    "picu": ["pediatric intensive care unit"],
    "icu": ["intensive care unit"],
    "ehr": ["emr", "electronic health record", "electronic medical record"],
    "dvt": ["deep vein thrombosis"],
    "tpa": ["tissue plasminogen activator", "thrombolytic"],
    "cad": ["coronary artery disease", "computer aided design"],
    "cam": ["computer aided manufacturing"],
    "cae": ["computer aided engineering"],
    "fea": ["finite element analysis"],
    "cfd": ["computational fluid dynamics"],
    "gd&t": ["geometric dimensioning and tolerancing", "gdt"],
    "cnc": ["computer numerical control"],
    "dfm": ["design for manufacturability", "design for manufacturing"],
    "dfmea": ["design failure mode and effects analysis"],
    "hvac": ["heating ventilation and air conditioning"],
    "plc": ["programmable logic controller"],
    "scada": ["supervisory control and data acquisition"],
    "pcb": ["printed circuit board"],
    "fpga": ["field programmable gate array"],
    "p&id": ["piping and instrumentation diagram"],
    "hazop": ["hazard and operability study"],
    "bim": ["building information modeling", "building information modelling"],
    "boq": ["bill of quantities"],
    "ros": ["robot operating system"],
    "slam": ["simultaneous localization and mapping"],
    "imu": ["inertial measurement unit"],
    "k8s": ["kubernetes"],
    "ci/cd": ["continuous integration continuous deployment", "cicd"],
    "iac": ["infrastructure as code"],
    "api": ["apis", "application programming interface"],
    "sdk": ["software development kit"],
    "rdbms": ["relational database"],
    "nosql": ["not only sql"],
    "orm": ["object relational mapping"],
    "sre": ["site reliability engineering"],
    "tdd": ["test driven development"],
    "bdd": ["behavior driven development"],
    "ssr": ["server side rendering"],
    "ui": ["user interface"],
    "ux": ["user experience"],
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "nlp": ["natural language processing"],
    "llm": ["large language model", "large language models"],
    "rag": ["retrieval augmented generation"],
    "genai": ["generative ai", "generative artificial intelligence"],
    "rl": ["reinforcement learning"],
    "rlhf": ["reinforcement learning from human feedback"],
    "eda": ["exploratory data analysis"],
    "bi": ["business intelligence"],
    "etl": ["extract transform load"],
    "elt": ["extract load transform"],
    "sem": ["scanning electron microscopy"],
    "tem": ["transmission electron microscopy"],
    "xrd": ["x-ray diffraction"],
    "afm": ["atomic force microscopy"],
    "nmr": ["nuclear magnetic resonance"],
    "hplc": ["high performance liquid chromatography"],
    "gc-ms": ["gas chromatography mass spectrometry"],
    "gis": ["geographic information system", "geographic information systems"],
    "iep": ["individualized education program"],
    "bip": ["behavioral intervention plan"],
    "aba": ["applied behavior analysis"],
    "lms": ["learning management system"],
    "cpa": ["certified public accountant"],
    "dcf": ["discounted cash flow"],
    "lbo": ["leveraged buyout"],
    "m&a": ["mergers and acquisitions"],
    "wacc": ["weighted average cost of capital"],
    "gaap": ["generally accepted accounting principles"],
    "ifrs": ["international financial reporting standards"],
    "sox": ["sarbanes oxley"],
    "ats": ["applicant tracking system"],
    "hris": ["human resources information system"],
}

# Spelling / wording variants of the SAME skill. First entry is the canonical form.
SYNONYM_GROUPS = [
    ["html", "html5", "html 5"],
    ["css", "css3", "css 3"],
    ["sql", "structured query language"],
    ["object oriented programming", "oop", "object oriented design", "ood", "object oriented concepts",
     "object oriented principles", "object oriented analysis and design", "ooad"],
    ["data structures and algorithms", "dsa", "data structures algorithms", "data structure and algorithm"],
    ["software development life cycle", "sdlc", "software development lifecycle"],
    ["rest api", "restful api", "restful web services", "restful services", "rest services", "rest web services"],
    ["microservices", "microservice", "microservices architecture", "microservice architecture"],
    ["node js", "nodejs"],
    ["react", "reactjs", "react js"],
    ["express js", "expressjs", "express"],
    ["vue js", "vuejs", "vue"],
    ["next js", "nextjs"],
    ["nest js", "nestjs"],
    ["angular", "angularjs", "angular js"],
    ["postgresql", "postgres", "postgre sql", "psql"],
    ["mongodb", "mongo", "mongo db"],
    ["mysql", "my sql"],
    ["microsoft sql server", "sql server", "mssql", "ms sql server"],
    ["javascript", "java script", "ecmascript"],
    ["typescript", "type script"],
    ["c#", "csharp", "c sharp"],
    ["c++", "cpp"],
    ["golang", "go lang"],
    ["amazon web services", "aws"],
    ["google cloud platform", "gcp", "google cloud"],
    ["microsoft azure", "azure"],
    ["scikit learn", "sklearn"],
    ["tensorflow", "tensor flow"],
    ["pytorch", "py torch"],
    ["html5", "html 5"],
    ["css3", "css 3"],
    ["unit testing", "unit test"],
    ["continuous integration", "ci"],
    ["github actions", "gh actions"],
    ["spring boot", "springboot"],
    ["power bi", "powerbi"],
    ["tailwind css", "tailwindcss", "tailwind"],
    ["responsive design", "responsive web design"],
    ["web socket", "websocket"],
    ["jira", "atlassian jira"],
]

_DROP_TERMS = {"ca", "go", "r", "c", "s", "t", "a"}
_SHORT_OK = {"ai", "ml", "bi", "ui", "ux", "dl", "rl", "rn", "ci", "cd", "ar", "vr", "qa"}

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+(?:[+#]+)?")


def _stem(t):
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("sses", "shes", "ches", "xes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def _tokens(text):
    return [_stem(t) for t in _TOKEN.findall((text or "").lower())]


def _norm_phrase(p):
    return " ".join(_tokens(p))


_CANON = {}
for _short, _exps in RAW_ACRONYM_MAP.items():
    _c = _norm_phrase(_short)
    for _v in [_short] + _exps:
        _CANON[_norm_phrase(_v)] = _c
for _grp in SYNONYM_GROUPS:
    _c = _norm_phrase(_grp[0])
    for _v in _grp:
        _CANON[_norm_phrase(_v)] = _c


def _key(raw):
    n = _norm_phrase(raw)
    n = _CANON.get(n, n)
    if not n or n in _DROP_TERMS:
        return None
    if " " not in n and len(n) <= 2 and n not in _SHORT_OK:
        return None
    return n


# --------------------------------------------------------------------------- #
# Professional skill display names ("rabbitmq" -> "RabbitMQ", "ci/cd" -> "CI/CD")
# --------------------------------------------------------------------------- #
_UPPER = set("""aws gcp api apis sql css html http https json xml yaml jwt orm dbms sdk ui ux sre iac etl elt nlp ml
ai dl rl llm rag ocr gpu cpu cuda gui cli ide oop dsa sdlc ssr csr spa pwa seo sem sea crm erp kpi kpis okr okrs roi
cac ltv ats hris sox gaap ifrs gst tds tcp udp dns cdn vpn vpc iam ec2 s3 rds eks aks gke ci cd cicd rpa sap abap bi
mis ecg ekg eeg emg mri pcr cpr bls acls pals nicu picu icu ehr emr hipaa gdpr fda ema iso osha cad cam cae fea cfd cnc
dfm fmea dfmea pfmea apqp ppap hvac plc scada pcb fpga rtos can lin uart spi usb adc dac pwm ros slam imu gis nmr hplc
xrd afm iot hdl vhdl gnc mbse hil sil qa sdet bdd tdd uat sit ssas ssrs ssis olap dax mdx lca bom ecn dcs pfd pid ndt
cmm cpa cma cfa frm rn lpn crna cra crc crf edc ctms tmf sdtm ind irb sae mdr kyc aml npa dscr ifr vfr atc cpl atpl
hr hris ar vr xr ide sso saml ldap siem soc edr xdr dlp ids ips waf ddos owasp nist cissp ceh oscp pci dss rest
ecs ecr emr sqs sns kms acl dom ajax bem wcag aria jsx tsx etc dba mvc mvvm mvi mvp ddd cqrs rpc grpc nlu asr tts""".split())
_UPPER |= {"solid"}
_UPPER -= {"rest", "etc", "can", "lin", "dom"}
_UPPER |= set("""aac aapm aba abb abr acas aci ada adam adas adb addie adhd adl adp afis aisc alara alm als apics apk apm apns
arima arr asa asc asce asme astm atls avr bds bert bim boq bsp bvsc cabg capa casa cbct cbse cch ccpa cdc cdisc cdo cdsco
cern cfe cibil cim cipd cism ciso ck clia cms cnn copd cpi cpt cre cro cscp csf csi ctf cvat cvd cve dag dao dast dcf
dcm dds dfa dfir dft dgca dicom dita dma dmd dna dpo easa ece ecm ecmo ecu eee ees ehs eia elisa elk emar emc emi emt
enps eoc epa eplan erdas esa esg esi etap eu ev evm faa fanuc fba fcas fcm fcpa fds fema fhir fia fico fmod fms fmva fsa
ftir gans garch gc gcih gcode gcs gdb glm glp glsl gmp gpio gps gpt grc gru gstr gtm haccp hazop hcm hec hft hie hlsl
hmi hms hpc hrbp hs hse iaea iatf ib ibnr icao icd ich icp idl idmt iec ieee iep ifoa igcse igloo ihc ii iii ilt imint
imrt inca ioc ipcc ipo ir iraf irc irr itil iv jpa jtag lasik lbo lc leed lidar lis lms loinc lora lstm lua lulc lut
lxd mcmc mcnp mds mips misra mitre mm moz mqtt mrp mrr ms mtm nabl ncert ncl nclex ndk ndma nepa ner nfpa ng ngo ngs
ngss nims niosh notam npv nrc nrp nsf nvh nwp nx obd obs onnx oos op opc orcid osint pacs pacu pbis pca pcl pde pdm
pecs peels peft pf pgt ph php pki pl plg plm pma pocus pp ppc ppe pr prd prt qpcr ras rasa rbt rcc rcra reba rfi rfid
rfp rfq rlhf rna rnn roas rpo rt rti rtl rtm rto rula saas sast sba sbar scorm scss sde sdgs sebi sla sli slo slp sme
soa soap sop sops ssg ssl stlc tavr taf tcas tia tiva tls tms tpm tpr ua udl uds ugc uplc ups usp utm uv vapt vba vfd
vfx vilt xapi xps xss yolo""".split())
_TOKEN_MAP_EXTRA = {
    "apis": "APIs", "rest": "REST", "admob": "AdMob", "arkit": "ARKit", "arxiv": "arXiv", "capiq": "CapIQ", "cmake": "CMake",
    "junit": "JUnit", "kicad": "KiCad", "spacy": "spaCy", "grads": "GrADS", "staad": "STAAD", "catia": "CATIA",
    "doors": "DOORS", "etabs": "ETABS", "rxjs": "RxJS", "ngrx": "NgRx", "nltk": "NLTK", "nmap": "Nmap", "mlops": "MLOps",
    "vllm": "vLLM", "qlora": "QLoRA", "okta": "Okta", "qlik": "Qlik", "xcode": "Xcode", "orcad": "OrCAD", "kuka": "KUKA",
    "hbase": "HBase", "hdfs": "HDFS", "scipy": "SciPy", "pnpm": "pnpm", "npm": "npm", "saas": "SaaS", "wifi": "Wi-Fi",
    "wincc": "WinCC", "lidar": "LiDAR", "tensorrt": "TensorRT", "opengl": "OpenGL", "webgl": "WebGL",
    "mapreduce": "MapReduce", "sagemaker": "SageMaker", "bigquery": "BigQuery", "devsecops": "DevSecOps",
    "gitops": "GitOps", "finops": "FinOps", "aiops": "AIOps", "sveltekit": "SvelteKit", "typeorm": "TypeORM",
    "mariadb": "MariaDB", "sqlite": "SQLite", "pytest": "pytest", "vue": "Vue", "wwise": "Wwise", "iot": "IoT",
    "pubsub": "Pub/Sub", "oauth": "OAuth", "openai": "OpenAI", "langchain": "LangChain", "pandas": "pandas",
}

_TOKEN_MAP = {
    "rabbitmq": "RabbitMQ", "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB", "nosql": "NoSQL",
    "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js", "express.js": "Express.js", "expressjs": "Express.js",
    "nestjs": "NestJS", "nest.js": "NestJS", "fastapi": "FastAPI", "github": "GitHub", "gitlab": "GitLab",
    "graphql": "GraphQL", "grpc": "gRPC", "oauth2": "OAuth2", "oauth": "OAuth", "restful": "RESTful", "devops": "DevOps",
    "typescript": "TypeScript", "javascript": "JavaScript", "asp.net": "ASP.NET", "tensorflow": "TensorFlow",
    "pytorch": "PyTorch", "ios": "iOS", "macos": "macOS", "powerbi": "Power BI", "sqlalchemy": "SQLAlchemy",
    "pyspark": "PySpark", "numpy": "NumPy", "openapi": "OpenAPI", "vue.js": "Vue.js", "next.js": "Next.js",
    "react.js": "React.js", "angular.js": "AngularJS", "d3.js": "D3.js", "three.js": "Three.js", "web3.js": "Web3.js",
    "ethers.js": "Ethers.js", "jquery": "jQuery", "html5": "HTML5", "css3": "CSS3", "k8s": "Kubernetes",
    "kubernetes": "Kubernetes", "jenkins": "Jenkins", "javafx": "JavaFX", "dynamodb": "DynamoDB", "cosmos": "Cosmos",
    "elasticsearch": "Elasticsearch", "opencv": "OpenCV", "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "huggingface": "Hugging Face", "mlflow": "MLflow", "tableau": "Tableau", "snowflake": "Snowflake", "dbt": "dbt",
    "airflow": "Airflow", "kafka": "Kafka", "redis": "Redis", "docker": "Docker", "terraform": "Terraform",
    "ansible": "Ansible", "grafana": "Grafana", "prometheus": "Prometheus", "datadog": "Datadog", "splunk": "Splunk",
    "wordpress": "WordPress", "youtube": "YouTube", "linkedin": "LinkedIn", "photoshop": "Photoshop",
    "autocad": "AutoCAD", "solidworks": "SolidWorks", "matlab": "MATLAB", "simulink": "Simulink", "labview": "LabVIEW",
    "solidity": "Solidity", "metamask": "MetaMask", "ipfs": "IPFS", "nft": "NFT", "defi": "DeFi", "dapp": "dApp",
    "erc20": "ERC-20", "erc721": "ERC-721", "erc1155": "ERC-1155", "wcag": "WCAG", "ci/cd": "CI/CD", "gd&t": "GD&T",
    "p&id": "P&ID", "m&a": "M&A", "eks": "EKS", "ec2": "EC2", "s3": "S3", "c++": "C++", "c#": "C#", "f#": "F#",
    "qgis": "QGIS", "arcgis": "ArcGIS", "nvivo": "NVivo", "spss": "SPSS", "sas": "SAS", "stata": "Stata",
    "powershell": "PowerShell", "bash": "Bash", "pagerduty": "PagerDuty", "servicenow": "ServiceNow",
    "salesforce": "Salesforce", "hubspot": "HubSpot", "netsuite": "NetSuite", "quickbooks": "QuickBooks",
    "xero": "Xero", "tally": "Tally", "gcp": "GCP", "aws": "AWS", "vscode": "VS Code", "iot": "IoT", "mern": "MERN",
    "mean": "MEAN", "lamp": "LAMP", "jamstack": "JAMstack", "sdlc": "SDLC", "dsa": "DSA", "oop": "OOP",
    "latex": "LaTeX", "ffmpeg": "FFmpeg", "fcp": "FCP", "unix": "Unix", "linux": "Linux", "ubuntu": "Ubuntu",
    "tcp/ip": "TCP/IP", "ui/ux": "UI/UX", "pwa": "PWA", "uat": "UAT", "sso": "SSO",
}
_TOKEN_MAP.update(_TOKEN_MAP_EXTRA)
_SMALL = {"and", "of", "for", "in", "to", "as", "on", "the", "with", "by", "or", "per", "vs"}
_SPLIT = re.compile(r"([/\-()&,])")


def pretty_skill(raw):
    s = (raw or "").strip()
    if not s:
        return s
    out = []
    for wi, w in enumerate(s.split()):
        lw = w.lower()
        if lw in _TOKEN_MAP:
            out.append(_TOKEN_MAP[lw])
            continue
        res = []
        for part in _SPLIT.split(w):
            lp = part.lower()
            if not part or part in "/-()&,":
                res.append(part)
            elif lp in _TOKEN_MAP:
                res.append(_TOKEN_MAP[lp])
            elif lp in _UPPER:
                res.append(lp.upper())
            elif lp in _SMALL and wi > 0:
                res.append(lp)
            else:
                res.append(lp[:1].upper() + lp[1:])
        out.append("".join(res))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Input validity gate
# --------------------------------------------------------------------------- #
_CODE_MARKERS = ("{%", "{{", "</", "/>", "function(", "function ", "=>", "const ", "let ", "var ", "return ",
                 "import ", "#include", "def ", "class ", "<div", "<span", "<script", "<style", "{\n", "};", "px;")


def _looks_like_code(text):
    """Pasted HTML / CSS / JS / templates must not be scored as a resume."""
    n = max(len(text), 1)
    sym = sum(text.count(c) for c in "{}<>;=$")
    markers = sum(text.count(m) for m in _CODE_MARKERS)
    return sym / n > 0.035 or markers >= 12


def is_valid_resume_content(text):
    clean = (text or "").strip()
    words = clean.split()
    if len(words) < 35:
        try:                                    # a short skills-only list is fine if it really is skills
            eng = get_engine()
            hits = sum(1 for j in eng.extract(clean) if eng.spec_w[j] > 0)
        except Exception:  # noqa: BLE001
            hits = 0
        if hits < 8:
            return False, "Input is insufficient for an ATS evaluation. Paste your full resume, or at least a list of 8+ skills."
    if _looks_like_code(clean):
        return False, "This looks like source code or markup, not a resume. Please paste your resume text."
    unique_words = set(w.lower() for w in words)
    if len(unique_words) / max(len(words), 1) < 0.30:
        return False, "Input has high repetition or lacks professional syntax."
    avg = sum(len(w) for w in words) / max(len(words), 1)
    if avg < 2.5 or avg > 18.0:
        return False, "Input appears to be randomized text."
    return True, None


def _header_text(text, max_lines=12):
    lines = [l.strip() for l in (text or "").splitlines() if len(l.strip()) > 3]
    return "\n".join(lines[:max_lines])


# --------------------------------------------------------------------------- #
# Scoring parameters
# --------------------------------------------------------------------------- #
CORE_REF_N, SUPP_REF_N = 18, 10    # a solid candidate shows ~14 core / 8 supporting skills of weight
CORE_SHARE = 0.75                  # competency = 75% core + 25% supporting
GAMMA = 0.90                       # concave curve: x**GAMMA (0.35 -> 43, 0.6 -> 66, 0.85 -> 88)
SCORE_CEIL = 96.0                  # nothing ever reaches 100
DISTINCT_W = 0.45                  # a skill counts as "distinctive" at/above this importance
TITLE_BONUS = 8.0                  # points for a job-title alias in the header (half for body mention)
REL_LO, REL_HI = 0.08, 0.40        # word-cosine -> relatedness (0.40+ = twin domains)
ENSEMBLE_W = 0.5
LEX_KAPPA = 10.0
TITLE_LOGIT = 1.5
DISPLAY_SKIP = {"database", "web technologies", "full stack", "full stack development", "documentation", "api"}
NOT_SKILLS = {"academic", "education", "web", "web technologies", "technologies", "tools", "programming", "development"}
AUTO_W = (0.70, 0.30)               # auto-detected role: closeness : enough-skills (70:30)
PREF_W = (0.65, 0.35)               # preferred role:     closeness : enough-skills
EVIDENCE_K = 0.9                    # pseudo-weight: very few skills cannot show a strong closeness
FAM_MIX = 0.25                      # share of closeness taken from the classifier's family probability


def hash_str(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return format(h, "08x")


class _Engine:
    def __init__(self, domains):
        self.domains = domains
        self.names = list(domains)
        N = len(self.names)
        entries = []                     # (dom, role, rank, list_len, key, raw)
        for i, name in enumerate(self.names):
            for role in ("aliases", "core", "supporting"):
                lst = domains[name].get(role, [])
                for r, raw in enumerate(lst):
                    k = _key(raw)
                    if k is not None:
                        entries.append((i, role, r, len(lst), k, raw))

        # merge keys that differ only by spacing ("node js"/"nodejs", "front end"/"frontend")
        groups = defaultdict(set)
        for e in entries:
            groups[e[4].replace(" ", "")].add(e[4])
        self._rep = {}
        self._alias_sq = {}
        for s, ks in groups.items():
            rep = max(ks, key=lambda k: (k.count(" "), k))
            for k in ks:
                self._rep[k] = rep
            if len(s) >= 6 or (len(s) >= 5 and " " in rep):
                self._alias_sq[s] = rep

        rep_keys = sorted(set(self._rep.values()))
        self.vocab = {k: j for j, k in enumerate(rep_keys)}
        V = len(rep_keys)
        self.V, self.N = V, N

        raws = defaultdict(list)
        for e in entries:
            raws[self._rep[e[4]]].append(e[5])
        from collections import Counter
        self.display = []
        for k in rep_keys:
            c = Counter(raws[k])
            self.display.append(pretty_skill(max(c, key=lambda r: (c[r], len(r)))))
        self.max_n = max(len(k.split()) for k in rep_keys)

        df = np.zeros(V)
        seen = set()
        for e in entries:
            j = self.vocab[self._rep[e[4]]]
            if (e[0], j) not in seen:
                seen.add((e[0], j))
                df[j] += 1
        denom = math.log(N) + 0.3
        self.imp = (np.log(N / df) + 0.3) / denom      # distinctiveness in (0.06, 1.0]
        self.feat_idf = 0.25 + np.log(N / df)           # classifier feature scaling

        core_w = np.zeros((N, V))
        supp_w = np.zeros((N, V))
        alias_m = np.zeros((N, V))
        order = {"aliases": [], "core": [], "supporting": []}
        self.dom_terms = [{"core": [], "supporting": [], "aliases": []} for _ in range(N)]
        for i, role, r, L, k, raw in entries:
            j = self.vocab[self._rep[k]]
            pos = 1.0 - 0.35 * (r / max(1, L - 1))
            w = self.imp[j] * pos
            if role == "core":
                core_w[i, j] = max(core_w[i, j], w)
            elif role == "supporting":
                supp_w[i, j] = max(supp_w[i, j], w)
            else:
                alias_m[i, j] = 1.0
            if j not in self.dom_terms[i][role]:
                self.dom_terms[i][role].append(j)
        supp_w[core_w > 0] = 0.0                         # a skill counts once, as core if listed there
        for i in range(N):
            self.dom_terms[i]["supporting"] = [j for j in self.dom_terms[i]["supporting"] if core_w[i, j] == 0]
        self.core_w, self.supp_w, self.alias_m = core_w, supp_w, alias_m
        self.member = ((core_w + supp_w) > 0).astype(float)          # skill belongs to domain (core or supporting)
        df = self.member.sum(axis=0)
        self.spec_w = np.where(df > 0, 1.0 / np.sqrt(np.maximum(df, 1.0)), 0.0)   # generic skills weigh less
        for j, nm in enumerate(self.display):                                       # filler words are not skills
            if _norm_phrase(nm) in NOT_SKILLS:
                self.spec_w[j] = 0.0
        self.ref_core = -np.sort(-core_w, axis=1)[:, :CORE_REF_N].sum(axis=1) + 1e-9
        self.ref_supp = -np.sort(-supp_w, axis=1)[:, :SUPP_REF_N].sum(axis=1) + 1e-9
        self.distinct = (np.maximum(core_w, supp_w) >= DISTINCT_W).astype(float)
        self._build_relatedness()
        self.signature = f"v3:{N}:{V}:{hash_str(''.join(rep_keys))}"

    def _build_relatedness(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = []
        for n in self.names:
            v = self.domains[n]
            docs.append(" ".join(v.get("aliases", []) * 2 + v.get("core", []) * 2 + v.get("supporting", [])).lower())
        tv = TfidfVectorizer(token_pattern=r"[a-z0-9+#]+", sublinear_tf=True, max_df=0.15, stop_words="english")
        X = tv.fit_transform(docs)
        cos = (X @ X.T).toarray()
        self.cos = cos
        rel = np.clip((cos - REL_LO) / (REL_HI - REL_LO), 0.0, 1.0)
        np.fill_diagonal(rel, 1.0)
        self.rel = rel

    # ---- extraction ------------------------------------------------------- #
    def extract(self, text):
        toks = _tokens(text)
        L = len(toks)
        found = set()
        rep, vocab, alias_sq = self._rep, self.vocab, self._alias_sq
        for i in range(L):
            t0 = toks[i]
            if len(t0) > 4 and t0.endswith("js") and " " not in t0:          # reactjs -> react js
                k = rep.get(_CANON.get(t0, t0)) or rep.get(_CANON.get(t0[:-2] + " js", t0[:-2] + " js"))
                if k:
                    found.add(vocab[k])
            for n in range(1, self.max_n + 1):
                if i + n > L:
                    break
                ph = " ".join(toks[i:i + n])
                ph = _CANON.get(ph, ph)
                k = rep.get(ph)
                if k is not None:
                    found.add(vocab[k])
                if 2 <= n <= 3:
                    sq = "".join(toks[i:i + n])
                    k2 = alias_sq.get(sq)
                    if k2 is not None:
                        found.add(vocab[k2])
        return found

    # ---- scoring ---------------------------------------------------------- #
    def features(self, body, head):
        from scipy.sparse import csr_matrix
        cols, vals = [], []
        for j in body:
            cols.append(j)
            vals.append(self.feat_idf[j])
        for j in head:
            cols.append(self.V + j)
            vals.append(1.5 * self.feat_idf[j])
        v = np.asarray(vals, dtype=float)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        return csr_matrix((v, ([0] * len(cols), cols)), shape=(1, 2 * self.V))

    def lexical(self, body, head):
        """Returns (raw, fit, p_lex, info). raw=matched importance, fit=competency/100."""
        x = np.zeros(self.V)
        h = np.zeros(self.V)
        if body:
            x[list(body)] = 1.0
        if head:
            h[list(head)] = 1.0
        mc, ms = self.core_w @ x, self.supp_w @ x
        core_cov = np.minimum(1.0, mc / self.ref_core)
        supp_cov = np.minimum(1.0, ms / self.ref_supp)
        core = 100.0 * core_cov ** GAMMA
        supp = 100.0 * supp_cov ** GAMMA
        skill = CORE_SHARE * core + (1 - CORE_SHARE) * supp
        dist = self.distinct @ x
        cap = np.select([dist == 0, dist == 1, dist == 2], [22.0, 38.0, 55.0], default=SCORE_CEIL)
        skill = np.minimum(skill, cap)
        core = np.minimum(core, cap)
        supp = np.minimum(supp, cap)
        title = np.where(self.alias_m @ h > 0, 1.0, np.where(self.alias_m @ x > 0, 0.5, 0.0))
        z = LEX_KAPPA * skill / 100.0 + TITLE_LOGIT * title
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        info = {"core": core, "supp": supp, "skill": skill, "title": title, "distinct": dist}
        return mc + ms, skill / 100.0, p, info


# --------------------------------------------------------------------------- #
# Lazy singletons
# --------------------------------------------------------------------------- #
_DOMAINS_CACHE = None
_ENGINE = None
_CLF = None
_CLF_LOADED = False


def load_domain_keywords():
    global _DOMAINS_CACHE
    if _DOMAINS_CACHE is None:
        try:
            with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
                _DOMAINS_CACHE = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"Error loading {DOMAIN_FILE}: {e}")
            _DOMAINS_CACHE = {}
    return _DOMAINS_CACHE


def get_domain_names():
    return list(load_domain_keywords().keys())


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _Engine(load_domain_keywords())
    return _ENGINE


def _load_classifier(engine):
    global _CLF, _CLF_LOADED
    if _CLF_LOADED:
        return _CLF
    _CLF_LOADED = True
    try:
        import joblib
        art = joblib.load(MODEL_FILE)
        if art.get("signature") == engine.signature and list(art["classes"]) == engine.names:
            _CLF = art
        else:
            print("domain_model.joblib is stale vs domain_keywords.json / matcher - "
                  "run train_domain_model.py. Using lexical scorer only.")
    except Exception:  # noqa: BLE001
        _CLF = None
    return _CLF


def analyze(resume_text):
    eng = get_engine()
    body = eng.extract(resume_text)
    head = eng.extract(_header_text(resume_text))
    raw, fit, p_lex, info = eng.lexical(body, head)
    art = _load_classifier(eng)
    if art is not None and body:
        p_clf = art["clf"].predict_proba(eng.features(body, head))[0]
        a = art.get("ensemble_w", ENSEMBLE_W)
        logp = a * np.log(p_clf + 1e-9) + (1 - a) * np.log(p_lex + 1e-9)
        logp -= logp.max()
        prob = np.exp(logp)
        prob /= prob.sum()
    else:
        p_clf, prob = None, p_lex

    skill = info["skill"]                       # sufficiency: do they have enough of the role's skills? (0-100)
    x = np.zeros(eng.V)
    if body:
        x[list(body)] = 1.0
    sx = eng.spec_w * x
    D = float(sx.sum())
    share = (eng.member @ sx) / (D + EVIDENCE_K)               # weighted share of the resume's skills inside each domain
    fam = np.minimum(1.0, eng.rel @ prob)
    closeness = (1.0 - FAM_MIX) * share + FAM_MIX * fam
    closeness = 1.0 - (1.0 - closeness) * (1.0 - 0.5 * info["title"])
    n_found = int(((eng.spec_w > 0) & (x > 0)).sum())
    n_in = (eng.member @ x).astype(int)
    wc, ws = AUTO_W
    align = wc * 100.0 * closeness + ws * skill
    align = np.minimum(align, 12.0 + 2.0 * skill)          # a role can't be rescued by profile alone
    align = np.where(info["distinct"] <= 0, np.minimum(align, 12.0), align)
    align = np.where(skill <= 0, 0.0, align)
    align = np.clip(align, 0.0, SCORE_CEIL)
    return {"engine": eng, "body": body, "head": head, "raw": raw, "fit": fit, "p_lex": p_lex,
            "p_clf": p_clf, "prob": prob, "skill": skill, "core": info["core"], "supp": info["supp"],
            "title": info["title"], "align": align, "closeness": closeness, "n_found": n_found, "n_in": n_in}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _domain_details(a, i, score):
    eng = a["engine"]
    body, head = a["body"], a["head"]
    terms = eng.dom_terms[i]
    weight = lambda j: max(eng.core_w[i, j], eng.supp_w[i, j])
    alias_ids = set(terms["aliases"])
    skip = lambda j: j in alias_ids or _norm_phrase(eng.display[j]) in DISPLAY_SKIP      # job titles / labels are not skills
    present_ids = sorted((j for j in terms["core"] + terms["supporting"] if j in body and not skip(j)), key=lambda j: -weight(j))
    names = [_norm_phrase(eng.display[j]) for j in present_ids]
    present_ids = [j for j, n in zip(present_ids, names)                                  # "Algorithms" is covered by "Data Structures and Algorithms"
                   if not any(n != m and (" " + n + " ") in (" " + m + " ") for m in names)]
    present = [eng.display[j] for j in present_ids]
    present_detail = [{"skill": eng.display[j], "type": "core" if eng.core_w[i, j] > 0 else "supporting"}
                      for j in present_ids]
    seen_stack = False
    miss_all = []
    for j in sorted((j for j in terms["core"] if j not in body and not skip(j)), key=lambda j: -eng.core_w[i, j]):
        if "stack" in eng.display[j].lower() or _norm_phrase(eng.display[j]) in ("mern", "mean", "lamp"):                                             # MERN / MEAN / LAMP ... show one only
            if seen_stack:
                continue
            seen_stack = True
        miss_all.append(j)
    miss_ids = miss_all[:10]
    n_high = max(1, math.ceil(len(miss_ids) * 0.4))
    missing_detail = [{"skill": eng.display[j], "priority": "high" if r < n_high else "medium"}
                      for r, j in enumerate(miss_ids)]
    title_hits = sum(1 for j in terms["aliases"] if j in head)
    return {
        "domain": eng.names[i],
        "score": int(round(score)),
        "raw_points": float(a["raw"][i]),
        "title_anchor_hits": title_hits,
        "present_skills": present,
        "present_detail": present_detail,
        "missing_skills": [m["skill"] for m in missing_detail],
        "missing_detail": missing_detail,
    }


def _family_mass(a, p):
    eng = a["engine"]
    return float(min(1.0, (eng.rel[p] * a["prob"]).sum()))


def detect_best_domain(resume_text):
    eng = get_engine()
    if not eng.names:
        return {"best_domain": "General Professional", "best_score": 50, "confident": False, "ranked_domains": []}

    valid, message = is_valid_resume_content(resume_text)
    empty = {"domain": "Unclassified / Insufficient Resume Content", "score": 0, "raw_points": 0,
             "title_anchor_hits": 0, "present_skills": [], "present_detail": [],
             "missing_skills": [], "missing_detail": []}
    if not valid:
        return {"best_domain": empty["domain"], "best_score": 0, "confident": False, "error": message,
                "ranked_domains": [], "best_details": empty}

    a = analyze(resume_text)
    if not a["body"] or a["raw"].max() <= 0 or a["align"].max() < 15 or a["n_found"] < 3:
        empty["domain"] = "Unclassified / No Matching Skills Detected"
        return {"best_domain": empty["domain"], "best_score": 0, "confident": False,
                "error": "We could not find enough recognisable skills. Paste your full resume, or at least a list of your tools, technologies and methods.",
                "ranked_domains": [], "best_details": empty}

    order = np.argsort(-a["align"], kind="stable")
    top = int(order[0])
    det = _domain_details(a, top, a["align"][top])
    second = float(a["align"][order[1]])
    top_s = float(a["align"][top])
    twin = bool(eng.rel[top, int(order[1])] >= 0.5)
    n_hits = len(det["present_skills"]) + det["title_anchor_hits"]
    confident = bool(top_s >= 45 and n_hits >= 3 and (top_s - second >= 6 or twin))

    ranked = [{"domain": eng.names[int(i)], "score": int(round(a["align"][int(i)])),
               "probability": round(float(a["prob"][int(i)]), 4)} for i in order[:6]]
    ranked = ([r for r in ranked if r["score"] >= 20] or ranked[:1])[:5]
    ranked.sort(key=lambda r: -r["score"])
    return {"best_domain": eng.names[top], "best_score": det["score"], "confident": confident,
            "confidence": round(_family_mass(a, top), 3), "ranked_domains": ranked, "best_details": det}


def _label(s):
    if s >= 78:
        return "Strong match"
    if s >= 60:
        return "Good match"
    if s >= 40:
        return "Partial match"
    if s >= 20:
        return "Weak match"
    return "Not a match"


FEW_SKILLS = 12       # fewer detected skills than this -> score is flagged as provisional


def _blend(a, p, weights):
    """Score for domain p: weights = (closeness share, sufficiency share)."""
    skill = float(a["skill"][p])
    sc = weights[0] * 100.0 * float(a["closeness"][p]) + weights[1] * skill
    sc = min(sc, 12.0 + 2.0 * skill)
    if a["skill"][p] <= 0:
        sc = 0.0
    return int(round(max(0.0, min(SCORE_CEIL, sc))))


def _evidence(a, p):
    """How much resume evidence backs the score. Thin resumes get a cautious score plus a clear notice."""
    n_in, n_all = int(a["n_in"][p]), int(a["n_found"])
    if n_in >= 10:
        level, msg = "strong", ""
    elif n_in >= 5:
        level, msg = "moderate", f"Only {n_in} skills for this field were found, so the score is a fair but cautious estimate."
    else:
        level = "limited"
        msg = (f"Only {n_in} skill{'s' if n_in != 1 else ''} for this field {'were' if n_in != 1 else 'was'} found"
               f" ({n_all} in total), which is too little to judge reliably. The score is cautious and may understate you - "
               "list your tools, frameworks and techniques in a dedicated Skills section.")
    return {"level": level, "skills_in_field": n_in, "skills_total": n_all, "message": msg}


def _reasons(a, p, det, score, weights, preferred):
    """Plain-language 'why' behind the score: strengths first, then gaps."""
    eng = a["engine"]
    name = eng.names[p]
    clos, suff = float(a["closeness"][p]) * 100.0, float(a["skill"][p])
    n_in, n_all = int(a["n_in"][p]), int(a["n_found"])
    out = []
    if n_all:
        if clos >= 70:
            out.append({"kind": "strength", "text": f"{n_in} of your {n_all} detected skills belong to this field, so your profile points clearly at it."})
        elif clos >= 45:
            out.append({"kind": "info", "text": f"{n_in} of your {n_all} detected skills belong to this field; the rest point to other fields."})
        else:
            out.append({"kind": "gap", "text": f"Only {n_in} of your {n_all} detected skills belong to this field, so your profile points mostly elsewhere."})
    if suff >= 65:
        out.append({"kind": "strength", "text": "You show enough of the field's key skills to be taken seriously for this role."})
    elif suff >= 35:
        out.append({"kind": "info", "text": "You show a fair share of the field's key skills, but important ones are still missing."})
    else:
        out.append({"kind": "gap", "text": "Too few of this field's key skills appear on your resume, which holds the score down."})
    miss = [m["skill"] for m in det["missing_detail"] if m["priority"] == "high"][:3]
    if miss and suff < 65:
        out.append({"kind": "gap", "text": "Adding " + ", ".join(miss) + " would lift this score the most."})
    top = int(np.argmax(a["align"]))
    if preferred and top != p:
        out.append({"kind": "info", "text": f"Your resume currently reads closer to {eng.names[top]} ({int(round(a['align'][top]))}%)."})
    return out


def match_score(resume_text, preferred_domain, mode=None):
    """Score (0-96) for a role.  Auto-detected role: 70% closeness + 30% sufficiency.
    Preferred role: 65% + 35%.  If the preferred role IS the detected role the auto formula is used,
    so both views show exactly the same result."""
    eng = get_engine()
    if preferred_domain not in eng.names:
        raise ValueError(f"Unknown domain: {preferred_domain!r}")
    a = analyze(resume_text)
    p = eng.names.index(preferred_domain)
    det = int(np.argmax(a["align"]))
    if mode is None:
        mode = "auto" if p == det else "preferred"
    weights = AUTO_W if mode == "auto" else PREF_W
    core, supp, skill = float(a["core"][p]), float(a["supp"][p]), float(a["skill"][p])
    closeness = float(a["closeness"][p])
    score = _blend(a, p, weights)
    details = _domain_details(a, p, skill)
    return {
        "score": score,
        "label": _label(score),
        "detected_domain": eng.names[det],
        "preferred_domain": preferred_domain,
        "mode": mode,
        "few_skills": bool(a["n_found"] < FEW_SKILLS),
        "skills_detected": int(a["n_found"]),
        "weights": {"closeness": weights[0], "sufficiency": weights[1]},
        "components": {
            "skill_closeness": round(closeness, 3),
            "skill_sufficiency": round(skill / 100.0, 3),
            "core_skill_match": round(core / 100.0, 3),
            "supporting_skill_match": round(supp / 100.0, 3),
        },
        "reasons": _reasons(a, p, details, score, weights, mode == "preferred"),
        "evidence": _evidence(a, p),
        "details": details,
        "related_domains": [eng.names[j] for j in np.argsort(-eng.rel[p])[1:4] if eng.rel[p, j] > 0.3],
    }


def screen_for_domain(resume_text, chosen_domain=None):
    """v1-compatible entry point; adds a 'match' block when a preferred role is given."""
    domains = load_domain_keywords()
    detection = detect_best_domain(resume_text)

    if "error" in detection or detection["best_score"] == 0:
        return {
            "domain": detection["best_domain"],
            "skills": {"coverage": 0, "present_skills": [], "missing_skills": [],
                       "present_detail": [], "missing_detail": []},
            "detection": {"confident": False, "best_score": 0, "ranked_domains": []},
            "match": {"score": 0, "label": "Not a match"},
            "error": detection.get("error"),
        }

    preferred = chosen_domain if chosen_domain and chosen_domain in domains else None
    active = preferred or detection["best_domain"]
    m = match_score(resume_text, active)
    coverage = m["score"]
    return {
        "domain": active,
        "skills": {
            "coverage": coverage,
            "present_skills": m["details"]["present_skills"],
            "missing_skills": m["details"]["missing_skills"],
            "present_detail": m["details"]["present_detail"],
            "missing_detail": m["details"]["missing_detail"],
        },
        "detection": {
            "confident": detection["confident"],
            "best_score": detection["best_score"],
            "detected_domain": detection["best_domain"],
            "ranked_domains": detection["ranked_domains"],
        },
        "match": {k: m[k] for k in ("score", "label", "detected_domain", "preferred_domain", "mode",
                                    "weights", "components", "reasons", "evidence", "few_skills", "skills_detected", "related_domains")},
    }


def warm_up():
    """Load the keyword engine and classifier once at import so the first (parallel) requests are fast and consistent."""
    try:
        _load_classifier(get_engine())
    except Exception as e:  # noqa: BLE001
        print(f"matcher warm-up skipped: {e}")


warm_up()