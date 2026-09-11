from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length


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
    submit = SubmitField("Send reset link")


class PasswordResetForm(FlaskForm):
    password1 = PasswordField("New password", validators=[DataRequired(), Length(min=8, message="Password must be at least 8 characters.")])
    password2 = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password1", message="Passwords must match.")])
    submit = SubmitField("Reset password")
