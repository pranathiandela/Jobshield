# JobShield — AUROS-based Fake Job Detection Starter

This version preserves the supplied Veridect landing-page design and adapts the branding to JobShield.

## Navbar
- Detector -> /detector
- Resume Screening -> /resume-screening
- Dashboard -> /dashboard
- History -> /history
- About & Help -> /about-help
- Get Started -> /login

## Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000

The detector currently uses transparent rule-based signals. It is intentionally structured so an ML model can replace/augment `analyze_job()` later.
