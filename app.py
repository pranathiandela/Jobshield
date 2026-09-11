from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import login_required
from pathlib import Path
from datetime import datetime
import json, re
from job_analyzer import analyze_job

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
    # Detector accepts multipart/form-data so the optional screenshot can be received
    # without changing the existing authentication or other project pages.
    data = request.form.to_dict(flat=True)
    data["recruiter_contact"] = data.get("recruiter_contact", "")
    screenshot = request.files.get("screenshot")
    if screenshot and screenshot.filename:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", screenshot.filename)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        screenshot.save(UPLOAD_DIR / f"detector_{stamp}_{safe_name}")

    result = analyze_job(data)
    history = load_history()
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": (data.get("job_title") or data.get("job_text", "").strip().splitlines() or ["Untitled job"])[0][:80],
        "company": data.get("company_name", ""),
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
