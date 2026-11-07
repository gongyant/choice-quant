from streamlit.testing.v1 import AppTest
at = AppTest.from_file(r'D:\choice_quant_starter\v2\p5_best_top5\dashboard\p5_app.py', default_timeout=60).run()
print('exceptions', [e.value for e in at.exception]); print('tables', len(at.dataframe))
