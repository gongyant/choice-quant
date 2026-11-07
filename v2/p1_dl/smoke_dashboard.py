"""Headless smoke test of stock_dashboard.py using streamlit.testing.v1.AppTest (no browser)."""
import sys, time
from pathlib import Path
from streamlit.testing.v1 import AppTest
H = Path(__file__).resolve().parent
def run_all():
    res = []; at = AppTest.from_file(str(H / "stock_dashboard.py"), default_timeout=180); t = time.time(); at.run()
    def ok(name):
        exc = [str(e.value) for e in at.exception]; res.append((name, not exc, exc[:1])); return not exc
    ok("page1 市场全览")
    for p in ["② 股票查询", "③ 单股页", "④ 模型对比", "⑤ 风险与数据质量"]:
        at.sidebar.radio(key="page").set_value(p).run(); ok("page " + p)
    # quick-access stocks on the stock page
    for c in ["600060", "300487", "000027"]:
        at.sidebar.button(key=f"quick_{c}").click().run(); ok(f"quick {c}: page={at.session_state['page']}, header={[h.value for h in at.header][:1]}")
    import pandas as pd
    lat = pd.read_csv(H / "latest_all_stocks_20261016.csv", dtype=str, encoding="utf-8-sig").set_index("code"); at.sidebar.radio(key="page").set_value("③ 单股页").run()
    for c in ["000006", "002829", "920699", "600485"]:
        at.selectbox(key="stk_select").set_value(f"{c} {lat.loc[c, 'name']}").run(); ok(f"select {c}: header={[h.value for h in at.header][:1]}")
    at.sidebar.radio(key="page").set_value("② 股票查询").run(); at.text_input(key="q").set_value("海信").run(); ok("search 海信")
    return res, round(time.time() - t, 1)
if __name__ == "__main__":
    res, secs = run_all()
    for r in res: print("PASS" if r[1] else "FAIL", r[0], r[2] if not r[1] else "")
    print("total", secs, "s"); sys.exit(0 if all(r[1] for r in res) else 1)
