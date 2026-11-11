
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    Regexp
)


# ==========================================
# STRONG PASSWORD VALIDATORS
# ==========================================

strong_password_validators = [

    DataRequired(
        message="Password is required."
    ),

    Length(
        min=8,
        message="Password must contain at least 8 characters."
    ),

    Regexp(
        r".*[A-Z].*",
        message="Password must contain at least one uppercase letter (A-Z)."
    ),

    Regexp(
        r".*[a-z].*",
        message="Password must contain at least one lowercase letter (a-z)."
    ),

    Regexp(
        r".*[0-9].*",
        message="Password must contain at least one number (0-9)."
    ),

    Regexp(
        r".*[^A-Za-z0-9].*",
        message="Password must contain at least one special character."
    )
]


# ==========================================
# SIGNUP FORM
# ==========================================

class SignupForm(FlaskForm):

    email = StringField(
        "Email address",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    password1 = PasswordField(
        "Password",
        validators=strong_password_validators
    )

    password2 = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo(
                "password1",
                message="Passwords must match."
            )
        ]
    )

    submit = SubmitField("Create account")


# ==========================================
# LOGIN FORM
# ==========================================

class LoginForm(FlaskForm):

    email = StringField(
        "Email address",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    password = PasswordField(
        "Password",
        validators=[
            DataRequired()
        ]
    )

    submit = SubmitField("Login")


# ==========================================
# USERNAME FORM
# ==========================================

class UsernameForm(FlaskForm):

    username = StringField(
        "Username",
        validators=[
            DataRequired(),
            Length(min=3, max=80)
        ]
    )

    submit = SubmitField("Continue")


# ==========================================
# PASSWORD RESET REQUEST FORM
# ==========================================

class PasswordResetRequestForm(FlaskForm):

    email = StringField(
        "Email address",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    submit = SubmitField("Send reset code")


# ==========================================
# OTP VERIFICATION FORM
# ==========================================

class OTPVerifyForm(FlaskForm):

    otp = StringField(
        "6-Digit Verification Code",
        validators=[
            DataRequired(),
            Length(
                min=6,
                max=6,
                message="Code must be exactly 6 digits."
            ),
            Regexp(
                r"^\d{6}$",
                message="Code must contain numbers only."
            )
        ]
    )

    submit = SubmitField("Verify Code")


# ==========================================
# PASSWORD RESET FORM
# ==========================================

class PasswordResetForm(FlaskForm):

    password1 = PasswordField(
        "New password",
        validators=strong_password_validators
    )

    password2 = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo(
                "password1",
                message="Passwords must match."
            )
        ]
    )

    submit = SubmitField("Update Password")
