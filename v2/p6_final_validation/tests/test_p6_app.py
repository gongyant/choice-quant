from streamlit.testing.v1 import AppTest
at=AppTest.from_file(r'D:\choice_quant_starter\v2\p6_final_validation\dashboard\p6_app.py', default_timeout=60).run()
print('exceptions',[str(e.value) for e in at.exception]); print('ok', len(at.dataframe))
