from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_mail import Message
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from extensions import db, mail
from forms import (
    LoginForm,
    PasswordResetForm,
    PasswordResetRequestForm,
    SignupForm,
    UsernameForm,
)
from google_auth import oauth
from models import User


auth_bp = Blueprint("auth", __name__)


def _serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="jobshield-password-reset")


def _make_reset_token(user):
    # Include the current password hash fingerprint so a used reset token
    # becomes invalid as soon as the password is changed.
    password_fingerprint = (user.password_hash or "")[-32:]
    return _serializer().dumps({"user_id": user.id, "password": password_fingerprint})


def _get_reset_user(token):
    try:
        data = _serializer().loads(token, max_age=current_app.config["PASSWORD_RESET_EXPIRY"])
        user = db.session.get(User, int(data["user_id"]))
        if not user:
            return None
        if data.get("password", "") != (user.password_hash or "")[-32:]:
            return None
        return user
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


def _safe_next(default_endpoint="main.home"):
    target = request.args.get("next") or request.form.get("next")
    if not target:
        return url_for(default_endpoint)
    parsed = urlparse(target)
    if parsed.netloc or parsed.scheme:
        return url_for(default_endpoint)
    return target if target.startswith("/") else url_for(default_endpoint)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = SignupForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        if User.query.filter_by(email=email).first():
            form.email.errors.append("An account with this email already exists.")
        else:
            user = User(email=email, needs_username=True)
            user.set_password(form.password1.data)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Account created. Just one more step.", "success")
            return redirect(url_for("auth.choose_username"))

    return render_template("signup.html", form=form, title="Create Account")


@auth_bp.route("/username", methods=["GET", "POST"])
@login_required
def choose_username():
    if not current_user.needs_username:
        return redirect(url_for("main.home"))

    form = UsernameForm(obj=current_user)
    if form.validate_on_submit():
        username = form.username.data.strip()
        existing = User.query.filter(User.username == username, User.id != current_user.id).first()
        if existing:
            form.username.errors.append("That username is already taken.")
        else:
            current_user.username = username
            current_user.needs_username = False
            db.session.commit()
            flash("Welcome to JobShield.", "success")
            return redirect(url_for("main.home"))

    return render_template("signup_username.html", form=form, title="Choose Username")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash("Welcome back.", "success")
            if user.needs_username:
                return redirect(url_for("auth.choose_username"))
            return redirect(_safe_next())
        flash("Invalid email or password.", "error")

    return render_template("login.html", form=form, title="Login")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("main.home"))


@auth_bp.route("/google/login")
def google_login():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))
    if not current_app.config.get("GOOGLE_CLIENT_ID") or not current_app.config.get("GOOGLE_CLIENT_SECRET"):
        flash("Google sign-in is not configured yet. You can use email and password.", "error")
        return redirect(url_for("auth.login"))
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = oauth.google.userinfo()
    except Exception:
        flash("Google sign-in could not be completed. Please try again.", "error")
        return redirect(url_for("auth.login"))

    email = (userinfo.get("email") or "").strip().lower()
    google_id = str(userinfo.get("sub") or "").strip()
    if not email or not google_id:
        flash("Google did not return the required account information.", "error")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()
        if user:
            user.google_id = google_id
        else:
            user = User(email=email, google_id=google_id, needs_username=True)
            db.session.add(user)
        db.session.commit()

    login_user(user)
    if user.needs_username:
        return redirect(url_for("auth.choose_username"))
    return redirect(url_for("main.home"))


@auth_bp.route("/password-reset", methods=["GET", "POST"])
def password_reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.strip().lower()).first()
        if user:
            token = _make_reset_token(user)
            reset_url = url_for("auth.password_reset_from_key", token=token, _external=True)

            # Send the reset link through the SMTP account configured in .env.
            # We deliberately keep the response generic so the page never reveals
            # whether an email address belongs to a JobShield account.
            if current_app.config.get("MAIL_USERNAME") and current_app.config.get("MAIL_PASSWORD"):
                try:
                    msg = Message(
                        subject="Reset your JobShield password",
                        recipients=[user.email],
                        sender=current_app.config.get("MAIL_DEFAULT_SENDER") or current_app.config.get("MAIL_USERNAME"),
                    )
                    msg.body = (
                        "We received a request to reset your JobShield password.\n\n"
                        f"Reset your password using this link:\n{reset_url}\n\n"
                        "This link expires in one hour. If you did not request this, you can ignore this email."
                    )
                    msg.html = f"""
                    <div style=\"font-family:Arial,sans-serif;line-height:1.6;color:#012624;max-width:600px;margin:auto\">
                      <h2 style=\"margin-bottom:8px\">Reset your JobShield password</h2>
                      <p>We received a request to reset your JobShield password.</p>
                      <p><a href=\"{reset_url}\" style=\"display:inline-block;padding:12px 20px;background:#edfffe;color:#012624;text-decoration:none;border:1px solid #9ed9d5;border-radius:8px;font-weight:600\">Reset password</a></p>
                      <p>This link expires in one hour. If you did not request this, you can ignore this email.</p>
                    </div>
                    """
                    mail.send(msg)
                    current_app.logger.info("Password reset email sent to %s", user.email)
                except Exception:
                    current_app.logger.exception("Password reset email failed")
            else:
                current_app.logger.warning(
                    "Password reset requested for %s, but MAIL_USERNAME/MAIL_PASSWORD are not configured.",
                    user.email,
                )
        flash("If an account exists for that email, a reset link has been sent.", "success")
        return redirect(url_for("auth.password_reset_done"))

    return render_template("password_reset.html", form=form, title="Reset Password")


@auth_bp.route("/password-reset/done")
def password_reset_done():
    return render_template("password_reset_done.html", title="Reset Email Sent")


@auth_bp.route("/password-reset/<token>", methods=["GET", "POST"])
def password_reset_from_key(token):
    user = _get_reset_user(token)
    if not user:
        flash("That password reset link is invalid or expired.", "error")
        return redirect(url_for("auth.password_reset_request"))

    form = PasswordResetForm()
    if form.validate_on_submit():
        user.set_password(form.password1.data)
        db.session.commit()
        flash("Your password has been reset. You can now log in.", "success")
        return redirect(url_for("auth.password_reset_from_key_done"))

    return render_template("password_reset_from_key.html", form=form, title="Set New Password")


@auth_bp.route("/password-reset/complete")
def password_reset_from_key_done():
    return render_template("password_reset_from_key_done.html", title="Password Updated")
