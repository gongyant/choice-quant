import zipfile, hashlib, re
z = zipfile.ZipFile("review_bundle.zip"); t = z.read("review_bundle/README_bundle.md").decode("utf-8")
rows = re.findall(r"\| `([^`]+)` \| .*? \| ([\d,]+) \| `([0-9a-f]{64})` \|", t); bad = 0
for a, n, h in rows:
    b = z.read("review_bundle/" + a)
    if hashlib.sha256(b).hexdigest() != h or len(b) != int(n.replace(",", "")): bad += 1; print("BAD", a)
print("rows", len(rows), "bad", bad)
for i in z.infolist(): print(f"{i.file_size:>11,} {i.compress_size:>10,} {i.filename}")
print("TOTAL zip", sum(i.compress_size for i in z.infolist()))
