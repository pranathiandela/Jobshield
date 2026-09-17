from datetime import datetime, timedelta, timezone
import secrets
from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_mail import Message

from extensions import db, mail
from forms import (
    LoginForm,
    OTPVerifyForm,
    PasswordResetForm,
    PasswordResetRequestForm,
    SignupForm,
    UsernameForm,
)
from google_auth import oauth
from models import User


auth_bp = Blueprint("auth", __name__)


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


# ============================================================================
# OTP PASSWORD RESET WORKFLOW (2-STEP)
# ============================================================================

@auth_bp.route("/password-reset", methods=["GET", "POST"])
def password_reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            otp_code = f"{secrets.randbelow(1_000_000):06d}"
            expiry_time = datetime.now(timezone.utc) + timedelta(minutes=10)
            user.set_reset_otp(otp_code, expiry_time)
            db.session.commit()

            session["reset_email"] = user.email
            session["otp_verified"] = False

            if current_app.config.get("MAIL_USERNAME") and current_app.config.get("MAIL_PASSWORD"):
                try:
                    msg = Message(
                        subject="Your JobShield Password Reset Code",
                        recipients=[user.email],
                        sender=current_app.config.get("MAIL_DEFAULT_SENDER") or current_app.config.get("MAIL_USERNAME"),
                    )
                    msg.body = (
                        f"Your JobShield verification code is: {otp_code}\n\n"
                        "This code is valid for 10 minutes.\n"
                        "If you did not request this, you can safely ignore this email."
                    )
                    msg.html = f"""
                    <div style="font-family:Arial,sans-serif;line-height:1.6;color:#012624;max-width:550px;margin:auto;padding:24px;border:1px solid #e0f2f1;border-radius:12px;">
                      <h2 style="margin-bottom:12px;color:#003734">Password Reset Code</h2>
                      <p>Enter the 6-digit verification code below to proceed with resetting your password:</p>
                      <div style="margin:24px 0;padding:16px;background:#edfffe;border:1px dashed #00827c;text-align:center;font-size:32px;font-weight:700;letter-spacing:6px;color:#012624;border-radius:8px">
                        {otp_code}
                      </div>
                      <p style="font-size:13px;color:#707777;">This code expires in 10 minutes. If you did not make this request, no action is needed.</p>
                    </div>
                    """
                    mail.send(msg)
                    current_app.logger.info("Password reset OTP sent to %s", user.email)
                except Exception:
                    current_app.logger.exception("Failed to send OTP email")
            else:
                current_app.logger.warning(
                    "MAIL credentials not configured. Development OTP for %s is %s",
                    user.email,
                    otp_code,
                )

        flash("If an account exists with that email, a 6-digit code has been sent.", "success")
        return redirect(url_for("auth.password_reset_verify"))

    return render_template("password_reset.html", form=form, title="Reset Password")


@auth_bp.route("/password-reset/verify", methods=["GET", "POST"])
def password_reset_verify():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    reset_email = session.get("reset_email")
    if not reset_email:
        flash("Please enter your email first.", "error")
        return redirect(url_for("auth.password_reset_request"))

    form = OTPVerifyForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=reset_email).first()
        entered_otp = form.otp.data.strip()

        if not user or not user.verify_reset_otp(entered_otp):
            flash("Invalid or expired code. Please verify and try again.", "error")
            return render_template("password_reset_verify.html", form=form, email=reset_email, title="Enter Verification Code")

        # Mark OTP as verified in session to unlock Step 2
        session["otp_verified"] = True
        flash("Code verified successfully. Now create your new password.", "success")
        return redirect(url_for("auth.password_reset_set_password"))

    return render_template("password_reset_verify.html", form=form, email=reset_email, title="Enter Verification Code")


@auth_bp.route("/password-reset/set-password", methods=["GET", "POST"])
def password_reset_set_password():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    reset_email = session.get("reset_email")
    otp_verified = session.get("otp_verified")

    # Gatekeep: Must have successfully passed OTP verification
    if not reset_email or not otp_verified:
        flash("Please verify your code first.", "error")
        return redirect(url_for("auth.password_reset_request"))

    form = PasswordResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=reset_email).first()
        if not user:
            flash("Account error. Please try again.", "error")
            return redirect(url_for("auth.password_reset_request"))

        user.set_password(form.password1.data)
        user.clear_reset_otp()
        db.session.commit()

        # Clean session
        session.pop("reset_email", None)
        session.pop("otp_verified", None)

        flash("Your password has been reset successfully. You can now log in.", "success")
        return redirect(url_for("auth.password_reset_complete"))

    return render_template("password_reset_from_key.html", form=form, title="Set New Password")


@auth_bp.route("/password-reset/complete")
def password_reset_complete():
    return render_template("password_reset_from_key_done.html", title="Password Updated")