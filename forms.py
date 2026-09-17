from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Regexp


class SignupForm(FlaskForm):
    email = StringField("Email address", validators=[DataRequired(), Email()])
    password1 = PasswordField("Password", validators=[DataRequired(), Length(min=8, message="Password must be at least 8 characters.")])
    password2 = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password1", message="Passwords must match.")])
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField("Email address", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class UsernameForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=80)])
    submit = SubmitField("Continue")


class PasswordResetRequestForm(FlaskForm):
    email = StringField("Email address", validators=[DataRequired(), Email()])
    submit = SubmitField("Send reset code")


# STEP 1: Verify OTP only
class OTPVerifyForm(FlaskForm):
    otp = StringField(
        "6-Digit Verification Code",
        validators=[
            DataRequired(),
            Length(min=6, max=6, message="Code must be exactly 6 digits."),
            Regexp(r"^\d{6}$", message="Code must contain numbers only.")
        ]
    )
    submit = SubmitField("Verify Code")


# STEP 2: Set new password
class PasswordResetForm(FlaskForm):
    password1 = PasswordField(
        "New password",
        validators=[DataRequired(), Length(min=8, message="Password must be at least 8 characters.")]
    )
    password2 = PasswordField(
        "Confirm password",
        validators=[DataRequired(), EqualTo("password1", message="Passwords must match.")]
    )
    submit = SubmitField("Update Password")