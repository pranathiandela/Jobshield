from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import login_required, current_user
from pathlib import Path
from datetime import datetime
import json, re
from job_analyzer import analyze_job
from resume_analyzer import analyze_resume
from ml_domain_matcher import screen_for_domain
from file_processor import extract_text, SUPPORTED_EXTENSIONS, UnsupportedFileType
from models import User, UserProfile, Scan

from werkzeug.utils import secure_filename
from config import Config
from extensions import db, login_manager, mail, csrf
from google_auth import init_google_oauth
from auth import auth_bp
from models import Scan

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
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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

    title = (data.get("job_title") or "Untitled job").strip()
    company = (data.get("company_name") or "").strip()

    scan = Scan(
        user_id=current_user.id,
        job_title=title[:255],
        company_name=company[:255],
        score=int(result.get("score", result.get("legitimacy_score", 0))),
        risk=str(result.get("risk", result.get("verdict", "Caution")))[:50],
        result_json=json.dumps(result, ensure_ascii=False),
    )
    db.session.add(scan)
    db.session.commit()

    result["scan_id"] = scan.id
    return jsonify(result)


@app.route("/resume-screening")
@login_required
def resume_screening():
    return render_template("resume.html", title="Resume Screening", active="resume")


@app.route("/api/screen-resume", methods=["POST"])
@login_required
@csrf.exempt
def api_screen_resume():
    resume_text = (request.form.get("resume_text") or "").strip()
    job_description = (request.form.get("job_description") or "").strip()
    resume_file = request.files.get("resume")

    if resume_file and resume_file.filename:
        ext = Path(resume_file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return jsonify({"error": "Please upload a PDF or DOCX file."}), 400

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", resume_file.filename)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_path = UPLOAD_DIR / f"resume_{stamp}_{safe_name}"
        resume_file.save(saved_path)

        try:
            resume_text = extract_text(saved_path)
        except UnsupportedFileType:
            return jsonify({"error": "Could not read that file type."}), 400

        if not resume_text:
            return jsonify({
                "error": "No readable text was found in that file. "
                         "Try pasting your resume text instead."
            }), 400

    if not resume_text:
        return jsonify({
            "error": "Paste your resume text or upload a PDF/DOCX file."
        }), 400

    # Analyze resume
    result = analyze_resume(resume_text, job_description or None)

    # -------------------------------
    # SAVE RESUME SCREENING TO DB
    # -------------------------------

    # Try to find whatever score your resume analyzer provides.
    score = (
        result.get("score")
        or result.get("match_score")
        or result.get("compatibility_score")
        or result.get("resume_score")
        or 0
    )

    try:
        score = int(float(score))
    except (TypeError, ValueError):
        score = 0

    score = max(0, min(100, score))

    # Convert resume score into the same risk categories
    # used by the Dashboard/History system.
    if score >= 70:
        risk = "Safe"
    elif score >= 40:
        risk = "Caution"
    else:
        risk = "Risky"

    scan = Scan(
        user_id=current_user.id,
        job_title="Resume Screening",
        company_name="",
        score=score,
        risk=risk,
        result_json=json.dumps(
            {
                "scan_type": "resume_screening",
                "result": result,
            },
            ensure_ascii=False
        ),
    )

    db.session.add(scan)
    db.session.commit()

    # Give frontend the database ID too
    result["scan_id"] = scan.id

    return jsonify(result)

@app.route("/api/screen-domain", methods=["POST"])
@login_required
@csrf.exempt
def api_screen_domain():
    resume_text = (request.form.get("resume_text") or "").strip()
    chosen_domain = (request.form.get("domain") or "").strip() or None
    resume_file = request.files.get("resume")

    if resume_file and resume_file.filename:
        ext = Path(resume_file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return jsonify({"error": "Please upload a PDF or DOCX file."}), 400
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", resume_file.filename)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_path = UPLOAD_DIR / f"domain_{stamp}_{safe_name}"
        resume_file.save(saved_path)
        try:
            resume_text = extract_text(saved_path)
        except UnsupportedFileType:
            return jsonify({"error": "Could not read that file type."}), 400
        if not resume_text:
            return jsonify({"error": "No readable text was found in that file."}), 400

    if not resume_text:
        return jsonify({"error": "Paste your resume text or upload a PDF/DOCX file."}), 400

    result = screen_for_domain(resume_text, chosen_domain)
    if not result:
        return jsonify({"error": "Could not screen this resume."}), 400

    return jsonify(result)


@app.route("/api/domain-list")
@login_required
def api_domain_list():
    """So the frontend can populate a manual domain-picker dropdown."""
    from ml_domain_matcher import get_domain_names
    return jsonify({"domains": sorted(get_domain_names())})

@app.route("/dashboard")
@login_required
def dashboard():
    scans = (
        Scan.query
        .filter_by(user_id=current_user.id)
        .order_by(Scan.created_at.desc())
        .all()
    )
    return render_template(
        "dashboard.html",
        title="Dashboard",
        active="dashboard",
        stats={
            "total": len(scans),
            "high": sum(x.risk == "Risky" for x in scans),
            "medium": sum(x.risk == "Caution" for x in scans),
            "low": sum(x.risk == "Safe" for x in scans),
        },
        recent_scans=scans[:5],
    )


@app.route("/history")
@login_required
def history():
    scans = (
        Scan.query
        .filter_by(user_id=current_user.id)
        .order_by(Scan.created_at.desc())
        .all()
    )
    return render_template(
        "history.html",
        title="History",
        active="history",
        history=scans,
    )


@app.route("/history/<int:scan_id>")
@login_required
def history_detail(scan_id):
    scan = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    result = scan.get_result()
    return render_template(
        "history.html",
        title="History",
        active="history",
        history=[scan],
        selected_scan=scan,
        selected_result=result,
    )

@app.route("/profile")
@login_required
def profile():
    profile = UserProfile.query.filter_by(
        user_id=current_user.id
    ).first()

    if not profile:
        profile = UserProfile(
            user_id=current_user.id,
            avatar_type="character",
            avatar_value="👤"
        )
        db.session.add(profile)
        db.session.commit()

    return render_template(
        "profile.html",
        title="My Profile",
        active="profile",
        profile=profile
    )
@app.route("/profile/update", methods=["POST"])
@login_required
@csrf.exempt
def profile_update():

    try:
        # Support JSON and normal form submissions
        data = request.get_json(silent=True)

        if data is None:
            data = request.form

        username = str(
            data.get("username", "")
        ).strip()

        if not username:
            return jsonify({
                "success": False,
                "error": "Username cannot be empty."
            }), 400

        if len(username) > 80:
            return jsonify({
                "success": False,
                "error": "Username must be 80 characters or less."
            }), 400

        # Check whether another user already has this username
        existing = User.query.filter(
            User.username == username,
            User.id != current_user.id
        ).first()

        if existing:
            return jsonify({
                "success": False,
                "error": "That username is already taken."
            }), 400

        # Update logged-in user
        current_user.username = username

        if hasattr(current_user, "needs_username"):
            current_user.needs_username = False

        db.session.commit()

        return jsonify({
            "success": True,
            "username": current_user.username
        }), 200

    except Exception as e:

        db.session.rollback()

        print(
            "PROFILE USERNAME ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": "Could not update username."
        }), 500


@app.route("/profile/password", methods=["POST"])
@login_required
@csrf.exempt
def profile_password():
    # Accept both JSON and normal form data
    data = request.get_json(silent=True) or request.form

    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""
    confirm_password = data.get("confirm_password") or ""

    # Google-only account
    if not current_user.password_hash:
        return jsonify({
            "success": False,
            "error": (
                "This account uses Google sign-in and does not "
                "have a JobShield password yet."
            )
        }), 400

    # Check current password
    if not current_user.check_password(current_password):
        return jsonify({
            "success": False,
            "error": "Current password is incorrect."
        }), 400

    # New password validation
    if len(new_password) < 8:
        return jsonify({
            "success": False,
            "error": "New password must be at least 8 characters."
        }), 400

    if new_password != confirm_password:
        return jsonify({
            "success": False,
            "error": "New passwords do not match."
        }), 400

    # Don't allow same password
    if current_user.check_password(new_password):
        return jsonify({
            "success": False,
            "error": "New password must be different from your current password."
        }), 400

    current_user.set_password(new_password)

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Password updated successfully."
    })


@app.route("/profile/avatar", methods=["POST"])
@login_required
@csrf.exempt
def profile_avatar():
    image = request.files.get("avatar")

    if image is None:
        return jsonify({
            "success": False,
            "error": "No image was received by the server."
        }), 400

    if not image.filename:
        return jsonify({
            "success": False,
            "error": "Please choose an image."
        }), 400

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }

    ext = Path(image.filename).suffix.lower()

    if ext not in allowed_extensions:
        return jsonify({
            "success": False,
            "error": "Please upload PNG, JPG, JPEG or WEBP."
        }), 400

    profile_dir = BASE / "static" / "uploads" / "profiles"
    profile_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = secure_filename(
        f"user_{current_user.id}{ext}"
    )

    save_path = profile_dir / filename

    try:
        # Save image
        image.save(str(save_path))

        # Verify file actually exists
        if not save_path.exists():
            return jsonify({
                "success": False,
                "error": "The image could not be saved."
            }), 500

        # Get profile
        profile = UserProfile.query.filter_by(
            user_id=current_user.id
        ).first()

        if not profile:
            profile = UserProfile(
                user_id=current_user.id,
                avatar_type="character",
                avatar_value="👤"
            )

            db.session.add(profile)

        # Save uploaded image information
        profile.avatar_type = "upload"
        profile.avatar_value = filename

        db.session.commit()

        return jsonify({
            "success": True,
            "avatar_url": url_for(
                "static",
                filename=f"uploads/profiles/{filename}"
            )
        })

    except Exception as e:
        db.session.rollback()

        print(
            "PROFILE IMAGE ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": "Could not save profile image."
        }), 500


@app.route("/profile/character", methods=["POST"])
@login_required
@csrf.exempt
def profile_character():
    # Accept both JSON and form data
    data = request.get_json(silent=True) or request.form

    character = (data.get("character") or "").strip()

    allowed_characters = {
        "👨‍💻",
        "👩‍💻",
        "🧑‍💻",
        "🤖",
        "🦊",
        "🐼",
        "🐱",
        "🐯",
        "🦁",
        "🐸",
        "🐨",
        "🐰",
        "🐻",
        "🦄",
        "👾",
        "🚀",
        "⭐",
        "👨",
        "👩",
        "🧑"
    }

    if character not in allowed_characters:
        return jsonify({
            "success": False,
            "error": "Invalid character."
        }), 400

    profile = UserProfile.query.filter_by(
        user_id=current_user.id
    ).first()

    if not profile:
        profile = UserProfile(
            user_id=current_user.id
        )

        db.session.add(profile)

    profile.avatar_type = "character"
    profile.avatar_value = character

    db.session.commit()

    return jsonify({
        "success": True,
        "character": character
    })

@app.route("/about-help")
def about_help():
    return render_template("help.html", title="About & Help", active="help")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
