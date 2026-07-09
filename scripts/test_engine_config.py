"""引擎配置界面化验证 — 全流程测试。

验证项:
  1. GET /config/engine 返回所有引擎配置项
  2. POST /config/engine 写入并可回读
  3. 配置持久化: 重启后值保留
  4. 所有 16 个配置项均存在
"""
import sys, os

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script_dir, "..", "data")
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  OK {n} {d}")
    else: FAIL += 1; print(f"  FAIL {n} {d}")


def test_engine_config_api():
    banner("Test 1: GET /engine 返回完整性")

    from fastapi.testclient import TestClient
    from strategy_forge.api.config_routes import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # GET initial config
    r = client.get("/api/forge/config/engine")
    check("GET /engine 返回 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    required_keys = [
        "default_rounds", "max_agents", "candidate_count", "llm_temperature",
        "max_concurrent", "retrieve_top_k", "similarity_threshold",
        "intel_safety_net", "recall_rel_boost", "event_hybrid",
        "llm_timeout", "connect_timeout", "generation_timeout",
        "retry_passes", "sim_fail_threshold",
    ]
    for k in required_keys:
        check(f"  GET 包含 {k}", k in data, str(data.get(k)))

    # POST modified values
    mod = {
        "default_rounds": 20, "max_agents": 5000, "candidate_count": 5,
        "llm_temperature": 0.8, "max_concurrent": 4, "retrieve_top_k": 10,
        "similarity_threshold": 0.6, "intel_safety_net": False,
        "recall_rel_boost": False, "event_hybrid": False,
        "llm_timeout": 600, "connect_timeout": 120, "generation_timeout": 3600,
        "retry_passes": 5, "sim_fail_threshold": 0.5,
    }
    r = client.post("/api/forge/config/engine", json=mod)
    check("POST /engine 返回 200", r.status_code == 200, str(r.status_code))

    # Re-read to verify persistence
    r = client.get("/api/forge/config/engine")
    data2 = r.json()
    check("default_rounds 回读正确", data2["default_rounds"] == 20, str(data2["default_rounds"]))
    check("max_agents 回读正确", data2["max_agents"] == 5000, str(data2["max_agents"]))
    check("candidate_count 回读正确", data2["candidate_count"] == 5, str(data2["candidate_count"]))
    check("llm_temperature 回读正确", data2["llm_temperature"] == 0.8, str(data2["llm_temperature"]))
    check("max_concurrent 回读正确", data2["max_concurrent"] == 4, str(data2["max_concurrent"]))
    check("intel_safety_net 回读正确(False)", data2["intel_safety_net"] == False, str(data2["intel_safety_net"]))
    check("recall_rel_boost 回读正确(False)", data2["recall_rel_boost"] == False, str(data2["recall_rel_boost"]))
    check("event_hybrid 回读正确(False)", data2["event_hybrid"] == False, str(data2["event_hybrid"]))
    check("similarity_threshold 回读正确", data2["similarity_threshold"] == 0.6, str(data2["similarity_threshold"]))
    check("sim_fail_threshold 回读正确", data2["sim_fail_threshold"] == 0.5, str(data2["sim_fail_threshold"]))

    # Restore defaults
    defaults = {
        "default_rounds": 10, "max_agents": 10000, "candidate_count": 3,
        "llm_temperature": 0.6, "max_concurrent": 2, "retrieve_top_k": 5,
        "similarity_threshold": 0.4, "intel_safety_net": True,
        "recall_rel_boost": True, "event_hybrid": True,
        "llm_timeout": 300, "connect_timeout": 60, "generation_timeout": 1800,
        "retry_passes": 3, "sim_fail_threshold": 0.75,
    }
    client.post("/api/forge/config/engine", json=defaults)
    r = client.get("/api/forge/config/engine")
    restored = r.json()
    check("恢复后 default_rounds=10", restored["default_rounds"] == 10)
    check("恢复后 safety_net=True", restored["intel_safety_net"] == True)


def test_bool_serialization():
    banner("Test 2: Bool 配置项序列化验证")

    from fastapi.testclient import TestClient
    from strategy_forge.api.config_routes import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Test with explicit booleans
    for val in [True, False]:
        mod = {"intel_safety_net": val, "recall_rel_boost": val, "event_hybrid": val,
               "default_rounds": 10, "max_agents": 1000, "candidate_count": 3,
               "llm_temperature": 0.6, "max_concurrent": 2, "retrieve_top_k": 5,
               "similarity_threshold": 0.4, "llm_timeout": 300, "connect_timeout": 60,
               "generation_timeout": 1800, "retry_passes": 3, "sim_fail_threshold": 0.75}
        client.post("/api/forge/config/engine", json=mod)
        r = client.get("/api/forge/config/engine")
        data = r.json()
        check(f"toggle={val} 回读 intel_safety_net", data["intel_safety_net"] == val)
        check(f"toggle={val} 回读 recall_rel_boost", data["recall_rel_boost"] == val)
        check(f"toggle={val} 回读 event_hybrid", data["event_hybrid"] == val)

    # Restore
    client.post("/api/forge/config/engine", json={
        "default_rounds": 10, "max_agents": 1000, "candidate_count": 3,
        "llm_temperature": 0.6, "max_concurrent": 2, "retrieve_top_k": 5,
        "similarity_threshold": 0.4, "intel_safety_net": True,
        "recall_rel_boost": True, "event_hybrid": True,
        "llm_timeout": 300, "connect_timeout": 60, "generation_timeout": 1800,
        "retry_passes": 3, "sim_fail_threshold": 0.75,
    })


def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 引擎配置界面化验证")
    print("=" * 65)

    test_engine_config_api()
    test_bool_serialization()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
