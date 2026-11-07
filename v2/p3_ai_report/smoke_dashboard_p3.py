# -*- coding: utf-8 -*-
"""Headless smoke test for the P3 dashboard (AppTest, no browser) + optional HTTP 200 check (run_dashboard_p3.ps1 -NoBrowser running on 8503)."""
import sys, time, os
from pathlib import Path
from streamlit.testing.v1 import AppTest
D = Path(__file__).resolve().parent / "dashboard"; os.chdir(D); sys.path.insert(0, str(D))
def run_all():
    res = []
    def ok(at, name): exc = [str(e.value) for e in at.exception]; res.append((name, not exc, exc[:1])); return not exc
    at = AppTest.from_file(str(D / "每日AI选股报告.py"), default_timeout=180); at.run(); ok(at, "main page")
    txt = " ".join(e.value for e in at.error); res.append(("disclaimer", "研究排名不是买入推荐" in txt, [txt[:60]]))
    at.radio(key="ver").set_value("v1(原规则)").run(); ok(at, "v1 toggle")
    at.radio(key="ver").set_value("v2(收紧趋势/防追高)").run(); ok(at, "v2 toggle")
    for q in ["600241", "德福", "688551", "zzzz"]: at.text_input(key="q").set_value(q).run(); ok(at, f"search {q}")
    at.text_input(key="q").set_value("000").run(); at.selectbox(key="code").set_value(at.selectbox(key="code").options[-1]).run(); ok(at, "search non-eligible/last")
    p = AppTest.from_file(str(D / "pages" / "1_历史预测vs实际.py"), default_timeout=180); p.run(); ok(p, "history page")
    for c in ["扣 0.30%", "扣 0.60%"]: p.selectbox(key="cost").set_value(c).run(); ok(p, f"cost {c}")
    p.radio(key="hz").set_value("20d").run(); ok(p, "hz 20d"); p.selectbox(key="strat").set_value("F3基线").run(); ok(p, "strat F3")
    return res
if __name__ == "__main__":
    t = time.time(); res = run_all()
    if "--http" in sys.argv:
        import urllib.request
        for u in ["http://127.0.0.1:8503/", "http://127.0.0.1:8503/_stcore/health"]:
            try: code = urllib.request.urlopen(u, timeout=10).status
            except Exception as e: code = str(e)
            res.append((f"HTTP {u}", code == 200, [code]))
    for r in res: print("PASS" if r[1] else "FAIL", r[0], r[2] if not r[1] else "")
    print("passed", sum(r[1] for r in res), "of", len(res), "secs", round(time.time() - t, 1)); sys.exit(0 if all(r[1] for r in res) else 1)