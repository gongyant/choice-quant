import zipfile,hashlib,glob,os
pats=['reports/*.md','results/*.csv','results/*.json','figures/*.png','*.py','*.ps1','*.json','*.jsonl','worklog.md','tests/*.py','dashboard/*.py','predictions_registry/*','preds/*_meta.json','cache/build_info.json','cache/fin_info.json']
fs=sorted({f for p in pats for f in glob.glob(p)})
sums=[hashlib.sha256(open(f,'rb').read()).hexdigest()+'  '+f for f in fs]
with zipfile.ZipFile('p4_review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in fs: z.write(f)
    z.writestr('SHA256SUMS.txt','\n'.join(sums)); z.writestr('README.txt','P4 Top5 quant review bundle (exploratory research, not investment advice). Excludes raw prices, raw financial data, feature caches and model weights. Main reports: reports/p4_model_comparison_report.md, reports/p4_top5_report.md')
print(len(fs),'files',os.path.getsize('p4_review_bundle.zip'),hashlib.sha256(open('p4_review_bundle.zip','rb').read()).hexdigest())
