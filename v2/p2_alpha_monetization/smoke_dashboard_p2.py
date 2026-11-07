# -*- coding: utf-8 -*-
"""Headless smoke test of stock_dashboard_p2.py via streamlit.testing.v1.AppTest (no browser): every page, key widgets switched."""
import sys, time
from pathlib import Path
from streamlit.testing.v1 import AppTest
H = Path(__file__).resolve().parent
def run_all():
    res = []; at = AppTest.from_file(str(H / "stock_dashboard_p2.py"), default_timeout=240); t = time.time(); at.run()
    def ok(name):
        exc = [str(e.value) for e in at.exception]; res.append((name, not exc, exc[:1])); return not exc
    ok("page 概览")
    pages = ["P2-A 信号解剖", "P2-B 成本感知组合", "P2-C Transformer 消融", "P1 逐股页面入口", "报告与图"]
    for p in pages: at.sidebar.radio(key="page").set_value(p).run(); ok("page " + p)
    at.sidebar.radio(key="page").set_value(pages[0]).run()
    for gv in at.selectbox(key="a_gv").options[:3]: at.selectbox(key="a_gv").set_value(gv).run(); ok(f"A group {gv}")
    at.selectbox(key="a_def").set_value("old_all").run(); ok("A decile old_all")
    at.sidebar.radio(key="page").set_value(pages[1]).run()
    for m, s in [("HGB", "C1_flat"), ("GRU", "C3_realistic"), ("Ridge", "C4_capacity_1e8")]: at.selectbox(key="b_m").set_value(m).run(); at.selectbox(key="b_s").set_value(s).run(); ok(f"B nav {m} {s}")
    at.sidebar.radio(key="page").set_value(pages[2]).run()
    for m in ["stability", "ann_net_c2", "alpha_c2_hac_t"]: at.selectbox(key="c_met").set_value(m).run(); at.selectbox(key="c_met2").set_value(m).run(); ok(f"C metric {m}")
    at.selectbox(key="c_scope").set_value("seed42").run(); ok("C scope seed42")
    at.sidebar.radio(key="page").set_value(pages[3]).run(); at.text_input(key="q").set_value("600").run(); ok("P1 lookup 600")
    at.sidebar.radio(key="page").set_value(pages[4]).run()
    for r in at.selectbox(key="rep").options: at.selectbox(key="rep").set_value(r).run(); ok(f"report {r}")
    at.selectbox(key="fig").set_value(at.selectbox(key="fig").options[-1]).run(); ok("figure last")
    at.sidebar.radio(key="page").set_value("研究选股器").run(); ok("page 研究选股器")
    txt = " ".join(e.value for e in at.error); res.append(("screener disclaimer", "不是买入推荐" in txt and "不是概率" in txt and "未被证明可在成本后变现" in txt, [txt[:80]]))
    at.number_input(key="s_liq").set_value(1.0).run(); ok("screener liquidity 1e8")
    at.slider(key="s_ntop").set_value(2).run(); ok("screener >=2 models top50")
    at.selectbox(key="s_sort").set_value("A2研究权重(六模型平均)").run(); ok("screener sort A2")
    at.checkbox(key="s_xac").check().run(); at.checkbox(key="s_xst").check().run(); at.checkbox(key="s_xlu").check().run(); ok("screener exclusions")
    if at.selectbox(key="s_code").options if [w for w in at.selectbox if w.key == "s_code"] else []:
        at.selectbox(key="s_code").set_value(at.selectbox(key="s_code").options[0]).run(); ok("screener stock detail")
        at.selectbox(key="s_hm").set_value("LGBM").run(); ok("screener realized LGBM")
    at.slider(key="s_ntop").set_value(6).run(); at.slider(key="s_std").set_value(0.01).run(); ok("screener empty filter")
    at.slider(key="s_ntop").set_value(0).run(); at.slider(key="s_std").set_value(0.5).run(); at.number_input(key="s_liq").set_value(0.0).run()
    for c in ("s_xac", "s_xst", "s_xlu"): at.checkbox(key=c).uncheck().run()
    ind = at.multiselect(key="s_ind").options[0]; at.multiselect(key="s_ind").select(ind).run(); at.multiselect(key="s_ex").select(at.multiselect(key="s_ex").options[0]).run(); ok(f"screener industry {ind}")
    return res, round(time.time() - t, 1)
if __name__ == "__main__":
    res, secs = run_all()
    for r in res: print("PASS" if r[1] else "FAIL", r[0], r[2] if not r[1] else "")
    print("total", secs, "s", "passed", sum(r[1] for r in res), "of", len(res)); sys.exit(0 if all(r[1] for r in res) else 1)