# -*- coding: utf-8 -*-
"""P2 dashboard tests: AppTest smoke over all pages/widgets; read-only source; launcher port 8502; P1 dashboards untouched."""
import sys, re, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
def test_apptest_smoke_all_pages():
    import smoke_dashboard_p2 as S
    res, _ = S.run_all(); bad = [r for r in res if not r[1]]; assert len(res) >= 20 and not bad, bad
def test_dashboard_is_read_only():
    src = (HERE / "stock_dashboard_p2.py").read_text(encoding="utf-8")
    for pat in (r"\.to_csv\(", r"\.to_parquet\(", r"write_text\(", r"open\([^)]*['\"]w", r"os\.remove", r"unlink\(", r"subprocess", r"requests\."): assert not re.search(pat, src), pat
def test_launcher_port():
    s = (HERE / "run_dashboard_p2.ps1").read_text(encoding="utf-8-sig"); assert "--server.port 8502" in s and "stock_dashboard_p2.py" in s and "127.0.0.1" in s
def test_p1_dashboards_untouched():
    t0 = dt.datetime(2026, 10, 19, 13, 23).timestamp(); p1 = HERE.parent / "p1_dl"
    for f in ("stock_dashboard.py", "stock_dashboard_v2.py", "run_dashboard.ps1", "run_dashboard_v1.ps1"): assert (p1 / f).stat().st_mtime < t0, f
def test_http_check_recorded():
    s = (HERE / "tests" / "http_dashboard_p2.txt").read_text(encoding="utf-8-sig"); assert "health HTTP 200" in s and "root HTTP 200" in s