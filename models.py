from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=True, index=True)
    password_hash = db.Column(db.String(255), nullable=True)
    google_id = db.Column(db.String(255), unique=True, nullable=True, index=True)
    needs_username = db.Column(db.Boolean, default=False, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Fields for OTP-based password reset
    reset_otp_hash = db.Column(db.String(255), nullable=True)
    reset_otp_expiry = db.Column(db.DateTime, nullable=True)

    scans = db.relationship(
        "Scan",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )

    profile = db.relationship(
        "UserProfile",
        backref="user",
        uselist=False,
        lazy=True,
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return bool(self.password_hash) and check_password_hash(
            self.password_hash,
            password
        )

    def set_reset_otp(self, otp_code, expiry_dt):
        self.reset_otp_hash = generate_password_hash(otp_code)
        # Store as naive UTC datetime for clean SQLite storage
        if expiry_dt.tzinfo is not None:
            expiry_dt = expiry_dt.astimezone(timezone.utc).replace(tzinfo=None)
        self.reset_otp_expiry = expiry_dt

    def verify_reset_otp(self, otp_code):
        if not self.reset_otp_hash or not self.reset_otp_expiry:
            return False
        # Compare current UTC against stored UTC
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if now_utc > self.reset_otp_expiry:
            return False
        return check_password_hash(self.reset_otp_hash, otp_code)

    def clear_reset_otp(self):
        self.reset_otp_hash = None
        self.reset_otp_expiry = None


class UserProfile(db.Model):
    __tablename__ = "user_profiles"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        unique=True,
        nullable=False,
        index=True
    )

    avatar_type = db.Column(
        db.String(20),
        nullable=False,
        default="character"
    )

    avatar_value = db.Column(
        db.String(255),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class Scan(db.Model):
    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    job_title = db.Column(
        db.String(255),
        nullable=False,
        default="Untitled job"
    )

    company_name = db.Column(
        db.String(255),
        nullable=True
    )

    score = db.Column(
        db.Integer,
        nullable=False,
        default=0
    )

    risk = db.Column(
        db.String(50),
        nullable=False,
        default="Caution"
    )

    result_json = db.Column(
        db.Text,
        nullable=False
    )

    def get_result(self):
        import json
        try:
            return json.loads(self.result_json)
        except (TypeError, ValueError):
            return {}


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None