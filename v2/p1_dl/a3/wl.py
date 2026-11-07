import sys, datetime as dt
msg = open(sys.argv[1], encoding="utf-8").read()
with open("worklog.md", "a", encoding="utf-8", newline="\n") as f: f.write(msg.rstrip("\n").replace("@@T@@", dt.datetime.now().strftime("%Y-%m-%d %H:%M")) + "\n")
