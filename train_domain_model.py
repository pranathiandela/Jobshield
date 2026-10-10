"""Train the JobShield domain classifier and evaluate the full matcher.

    python train_domain_model.py                  # train + evaluate + save domain_model.joblib
    python train_domain_model.py --extra data.csv # also learn from real labelled resumes
                                                  # (columns: text, domain  - domain must be
                                                  #  one of the names in domain_keywords.json)

Training data = synthetic resumes generated from domain_keywords.json (realistic
layouts, acronym variants, career-switcher noise, thin resumes, missing titles),
plus any real labelled resumes passed with --extra. Real data is strongly
recommended: synthetic-only results are an upper bound.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression

import ml_domain_matcher as M

OUT = Path(__file__).resolve().parent / "domain_model.joblib"

SOFT = ["team player", "strong communication skills", "problem solving", "leadership",
        "time management", "detail oriented", "self-motivated", "collaboration",
        "critical thinking", "adaptable", "fast learner", "stakeholder management"]
FILLER = ["responsible for", "worked closely with cross-functional teams to deliver",
          "improved efficiency by 20 percent", "managed multiple priorities", "mentored junior colleagues",
          "contributed to quarterly goals", "presented findings to leadership",
          "participated in code and design reviews", "supported day to day operations",
          "delivered projects on schedule", "documented processes and outcomes"]
FIRST = ["Alex", "Priya", "Sam", "Maria", "Chen", "Ravi", "Olivia", "Daniel", "Aisha", "Lucas"]
LAST = ["Sharma", "Johnson", "Kumar", "Garcia", "Lee", "Patel", "Brown", "Nguyen", "Khan", "Miller"]
EDU = ["B.Tech", "B.Sc", "M.Sc", "MBA", "B.E.", "M.Tech", "Bachelor of Science", "Master's degree"]


class Generator:
    def __init__(self, engine, seed):
        self.e = engine
        self.rng = random.Random(seed)
        self.variants = {}
        for k, c in M._CANON.items():
            self.variants.setdefault(c, set()).add(k)

    def _vary(self, j, p):
        """Maybe swap a term for an acronym/expansion variant."""
        term = self.e.display[j]
        if self.rng.random() < p:
            canon = M._CANON.get(M._norm_phrase(term), M._norm_phrase(term))
            v = list(self.variants.get(canon, []))
            if len(v) > 1:
                return self.rng.choice(v)
        return term

    def resume(self, i, hard=False):
        r, e = self.rng, self.e
        t = e.dom_terms[i]
        if hard:
            nc, ns, p_switch, p_title, p_var = r.randint(2, 9), r.randint(0, 6), 0.6, 0.45, 0.4
        else:
            nc, ns, p_switch, p_title, p_var = r.randint(4, 15), r.randint(1, 9), 0.3, 0.75, 0.25
        core = r.sample(t["core"], min(nc, len(t["core"])))
        sup = r.sample(t["supporting"], min(ns, len(t["supporting"]))) if t["supporting"] else []
        terms = [self._vary(j, p_var) for j in core + sup]

        noise = []
        if r.random() < p_switch:
            if r.random() < 0.6:   # neighbour domain (career switch / adjacent skills)
                cand = np.argsort(-e.rel[i])[1:7]
                o = int(r.choice(list(cand)))
            else:
                o = r.randrange(e.N)
            if o != i:
                pool = e.dom_terms[o]["core"] + e.dom_terms[o]["supporting"]
                noise = [self._vary(j, p_var) for j in
                         r.sample(pool, min(r.randint(1, 7 if hard else 5), len(pool)))]
        terms_all = terms + noise
        r.shuffle(terms_all)

        title = ""
        if r.random() < p_title and t["aliases"]:
            title = e.display[r.choice(t["aliases"])].title()
        lines = [f"{r.choice(FIRST)} {r.choice(LAST)}"]
        if title:
            lines.append(title)
        lines.append("email@example.com | +1 555 0100 | linkedin.com/in/someone")
        half = max(1, len(terms_all) // 3)
        lines += ["SUMMARY",
                  f"{r.choice(SOFT).capitalize()} professional with experience in "
                  + ", ".join(terms_all[:half]) + ". " + r.choice(FILLER) + "."]
        lines += ["SKILLS", ", ".join(terms_all[half:2 * half] + r.sample(SOFT, 2))]
        lines.append("EXPERIENCE")
        for chunk in range(2 * half, len(terms_all), 3):
            ch = terms_all[chunk:chunk + 3]
            lines.append(f"- {r.choice(FILLER)} using " + " and ".join(ch) + ".")
        lines.append(f"- {r.choice(FILLER)}; {r.choice(FILLER)}.")
        lines += ["EDUCATION", f"{r.choice(EDU)} - University of {r.choice(LAST)}"]
        return "\n".join(lines)


def make_set(engine, per_domain, seed, hard):
    g = Generator(engine, seed)
    texts, ys = [], []
    for i in range(engine.N):
        for _ in range(per_domain):
            texts.append(g.resume(i, hard))
            ys.append(i)
    return texts, np.array(ys)


def featurize(engine, texts):
    rows, lex = [], []
    for tx in texts:
        body, head = engine.extract(tx), engine.extract(M._header_text(tx))
        rows.append(engine.features(body, head))
        lex.append(engine.lexical(body, head))
    return vstack(rows).tocsr(), lex


def evaluate(engine, clf, texts, y, lex, X, a_ens, label):
    P_clf = clf.predict_proba(X)
    res = {}
    rel = engine.rel
    for name, a in (("lexical only", 0.0), ("classifier only", 1.0), (f"ensemble a={a_ens}", a_ens)):
        top1 = top3 = fam = 0
        for n in range(len(y)):
            p_lex = lex[n][2]
            if a == 0.0:
                p = p_lex
            elif a == 1.0:
                p = P_clf[n]
            else:
                lp = a * np.log(P_clf[n] + 1e-9) + (1 - a) * np.log(p_lex + 1e-9)
                p = np.exp(lp - lp.max())
            order = np.argsort(-p)
            top1 += order[0] == y[n]
            top3 += y[n] in order[:3]
            fam += rel[y[n], order[0]] >= 0.5      # right domain or a twin of it
        res[name] = (top1 / len(y), top3 / len(y), fam / len(y))
    print(f"\n[{label}]  n={len(y)}   top-1 | top-3 | top-1 incl. twin domains")
    for k, v in res.items():
        print(f"  {k:22} {v[0]:.1%} | {v[1]:.1%} | {v[2]:.1%}")
    return res


def tune_ensemble(engine, clf, X, y, lex):
    P = clf.predict_proba(X)
    best = (-1, 0.5)
    for a in (0.2, 0.35, 0.5, 0.65, 0.8):
        ok = 0
        for n in range(len(y)):
            lp = a * np.log(P[n] + 1e-9) + (1 - a) * np.log(lex[n][2] + 1e-9)
            ok += int(np.argmax(lp) == y[n])
        acc = ok / len(y)
        if acc > best[0]:
            best = (acc, a)
    print(f"tuned ensemble weight a={best[1]} (val top-1 {best[0]:.1%})")
    return best[1]


def load_extra(path, engine):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df["domain"].isin(engine.names)].dropna(subset=["text"])
    print(f"extra real data: {len(df)} usable rows")
    return list(df["text"]), np.array([engine.names.index(d) for d in df["domain"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=90)
    ap.add_argument("--extra", default=None)
    args = ap.parse_args()

    eng = M.get_engine()
    print(f"{eng.N} domains, {eng.V} distinct skills/titles in vocabulary")
    t0 = time.time()
    tr_t, tr_y = make_set(eng, args.per_domain, seed=1, hard=False)
    hd_t, hd_y = make_set(eng, args.per_domain // 3, seed=2, hard=True)      # harder training mix
    tr_t, tr_y = tr_t + hd_t, np.concatenate([tr_y, hd_y])
    if args.extra:
        et, ey = load_extra(args.extra, eng)
        tr_t, tr_y = tr_t + et * 3, np.concatenate([tr_y, np.tile(ey, 3)])   # up-weight real data
    va_t, va_y = make_set(eng, 12, seed=3, hard=True)                        # validation (tuning)
    te_t, te_y = make_set(eng, 20, seed=99, hard=False)                      # test: normal
    th_t, th_y = make_set(eng, 20, seed=100, hard=True)                      # test: hard

    Xtr, _ = featurize(eng, tr_t)
    print(f"featurized {len(tr_t)} training resumes in {time.time() - t0:.0f}s; fitting...")
    clf = LogisticRegression(C=30.0, max_iter=400, solver="lbfgs")
    clf.fit(Xtr, tr_y)
    print(f"fit done in {time.time() - t0:.0f}s")

    Xva, lva = featurize(eng, va_t)
    a = tune_ensemble(eng, clf, Xva, va_y, lva)
    Xte, lte = featurize(eng, te_t)
    Xth, lth = featurize(eng, th_t)
    evaluate(eng, clf, te_t, te_y, lte, Xte, a, "TEST - normal resumes")
    evaluate(eng, clf, th_t, th_y, lth, Xth, a, "TEST - hard resumes (thin, no title, 60% cross-domain noise)")

    joblib.dump({"clf": clf, "classes": eng.names, "signature": eng.signature,
                 "ensemble_w": a, "synthetic_only": args.extra is None}, OUT, compress=3)
    print(f"\nsaved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
