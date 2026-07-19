"""E2E test via run.py server + LM Studio. 10 rounds, 维东国之变."""
import subprocess, time, requests, sys, os

API = "http://127.0.0.1:8000/api/forge"
TEXT = r"E:\gongxiang\软件\资本论\维东国之变.txt"

proc = subprocess.Popen([sys.executable, "run.py", "--port", "8000"],
    cwd=r"E:\gongxiang\StrategyForge", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(12)

try:
    r = requests.get(f"{API}/domains", timeout=5); r.raise_for_status()
    source = open(TEXT, encoding="utf-8").read()
    t0 = time.time()
    r = requests.post(f"{API}/session", json={"title":"test","source_material":source,"config":{"domain":"narrative","total_rounds":10}})
    sid = r.json()["id"]
    requests.post(f"{API}/config/engine", json={"max_concurrent":2})
    requests.post(f"{API}/session/{sid}/start")
    last_round = 0
    while True:
        time.sleep(10)
        s = requests.get(f"{API}/session/{sid}").json()
        cr = s.get("current_round", 0)
        st = s.get("status", "")
        if cr != last_round:
            print(f"  R{cr}/10  status={st}  t={time.time()-t0:.0f}s", flush=True)
            last_round = cr
        if st in ("complete", "failed", "paused"): break
    elapsed = time.time() - t0
    s = requests.get(f"{API}/session/{sid}").json()
    print(f"\nStatus={s['status']} rounds={s['current_round']}/{s['total_rounds']} agents={s.get('agent_count',0)} time={elapsed:.0f}s")
    if s.get("error"): print(f"Error: {s['error'][:200]}")
    t = requests.get(f"{API}/session/{sid}/tokens").json().get("stats",{})
    print(f"Tokens: {t.get('total_tokens',0)}")
    print("PASS" if s['status']=='complete' else "NOTE: not complete")
finally:
    proc.terminate(); proc.wait()
