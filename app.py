from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import login_required
from pathlib import Path
from datetime import datetime
import json, re

from config import Config
from extensions import db, login_manager, mail, csrf
from google_auth import init_google_oauth
from auth import auth_bp

app = Flask(__name__)
app.config.from_object(Config)

# Authentication services

db.init_app(app)
login_manager.init_app(app)
mail.init_app(app)
csrf.init_app(app)
init_google_oauth(app)
app.register_blueprint(auth_bp)

BASE = Path(__file__).resolve().parent
HISTORY_FILE = BASE / "history.json"
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(items):
    HISTORY_FILE.write_text(json.dumps(items[-100:], indent=2))


def analyze_job(text="", url="", email=""):
    text_l = (text or "").lower()
    signals = []
    score = 0
    rules = [
        (["pay a fee", "registration fee", "processing fee", "training fee", "deposit", "pay upfront"], 30, "Requests money, a fee, deposit, or upfront payment."),
        (["whatsapp only", "telegram only", "contact me on whatsapp", "contact via telegram"], 18, "Pushes communication to an informal channel instead of a normal hiring process."),
        (["guaranteed job", "guaranteed placement", "100% selected", "instant joining"], 16, "Uses unusually strong guarantees about hiring or selection."),
        (["no interview", "without interview"], 18, "Suggests hiring without a normal interview process."),
        (["crypto", "bitcoin", "usdt"], 20, "Mentions cryptocurrency in a hiring/payment context."),
        (["bank account", "otp", "one time password", "credit card", "debit card"], 25, "Requests sensitive financial/account information."),
        (["earn $", "earn ₹", "rs.", "per day"], 5, "Compensation language may need additional verification."),
    ]
    for words, points, message in rules:
        if any(w in text_l for w in words):
            score += points
            signals.append(message)
    if url:
        if re.search(r"https?://", url) is None:
            score += 8
            signals.append("The supplied job URL does not look like a normal web URL.")
        if any(x in url.lower() for x in [".tk", ".ml", ".ga", ".cf", "bit.ly", "tinyurl"]):
            score += 15
            signals.append("The URL uses a pattern that deserves extra verification.")
    if email:
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            score += 8
            signals.append("Recruiter email format looks unusual.")
        elif any(x in email.lower() for x in ["gmail.com", "yahoo.com", "outlook.com", "proton.me"]):
            score += 7
            signals.append("Recruiter uses a generic email provider; verify the company independently.")
    score = min(score, 100)
    risk = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    summary = {
        "High": "Several strong warning signals were detected. Verify the recruiter and company independently before sharing money or sensitive information.",
        "Medium": "Some warning signals were detected. Proceed carefully and verify the employer through an official company source.",
        "Low": "Few rule-based red flags were detected. A low score is not proof that a job is legitimate."
    }[risk]
    return {"score": score, "risk": risk, "signals": signals, "summary": summary}


@app.route("/", endpoint="main.home")
def home():
    return render_template("home.html")


@app.route("/detector")
@login_required
def detector():
    return render_template("detector.html", title="Detector", active="detector")


@app.route("/api/analyze-job", methods=["POST"])
@login_required
@csrf.exempt
def api_analyze():
    data = request.get_json(silent=True) or {}
    result = analyze_job(data.get("job_text", ""), data.get("job_url", ""), data.get("recruiter_email", ""))
    history = load_history()
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": (data.get("job_text", "").strip().splitlines() or ["Untitled job"])[0][:80],
        "risk": result["risk"],
        "score": result["score"]
    })
    save_history(history)
    return jsonify(result)


@app.route("/resume-screening", methods=["GET", "POST"])
@login_required
def resume_screening():
    result = None
    if request.method == "POST":
        f = request.files.get("resume")
        if not f or not f.filename:
            result = {"status": "Needs attention", "message": "Please select a resume file.", "signals": []}
        else:
            ext = Path(f.filename).suffix.lower()
            if ext not in {".pdf", ".doc", ".docx"}:
                result = {"status": "Unsupported", "message": "Please upload PDF, DOC, or DOCX.", "signals": []}
            else:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", f.filename)
                f.save(UPLOAD_DIR / safe_name)
                result = {
                    "status": "Screened",
                    "message": "The resume file was received successfully. Advanced ML exposure matching can be connected next.",
                    "signals": ["File type accepted.", "Resume is stored in the local project uploads folder.", "No external exposure database is queried by this starter version."]
                }
    return render_template("resume.html", title="Resume Screening", active="resume", result=result)


@app.route("/dashboard")
@login_required
def dashboard():
    h = load_history()
    return render_template("dashboard.html", title="Dashboard", active="dashboard", stats={
        "total": len(h),
        "high": sum(x.get("risk") == "High" for x in h),
        "medium": sum(x.get("risk") == "Medium" for x in h),
        "low": sum(x.get("risk") == "Low" for x in h)
    })


@app.route("/history")
@login_required
def history():
    return render_template("history.html", title="History", active="history", history=load_history())


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", title="My Profile", active="profile")


@app.route("/about-help")
def about_help():
    return render_template("help.html", title="About & Help", active="help")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
