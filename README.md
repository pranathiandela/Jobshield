# JobShield

AUROS-style JobShield landing page with authentication integrated.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 app.py
```

Open `http://127.0.0.1:5000/`.

### Authentication

- `/login` — login
- `/signup` — create account
- `/username` — choose username after signup
- `/password-reset` — password reset request
- `/logout` — logout
- Google OAuth is optional and requires credentials in `.env`.
- Detector, Resume Screening, Dashboard and History require login.

The existing AUROS landing page and existing JobShield pages are kept intact; authentication is added around them.


## Google Sign-In setup

The login page already includes **Continue with Google**. To enable the real OAuth flow:

1. Create/select a Google Cloud project.
2. Configure the OAuth consent screen.
3. Create an OAuth 2.0 Client ID of type **Web application**.
4. Add this Authorized redirect URI for local development:
   `http://127.0.0.1:5000/google/callback`
5. Put the client ID and secret in `.env` as `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.
6. Restart Flask and use **Continue with Google** on `/login`.

For a first-time Google account, JobShield sends the user to the username page. After a username is chosen, the user returns to the JobShield landing page.

## Account menu

After login, the username in the navbar opens a dropdown containing **My Profile** and **Logout**. Before login, **Get Started** uses the AUROS aurora gradient and opens the login page.
