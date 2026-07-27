"""测试 LM Studio 是否支持 response_format (JSON Schema / grammar sampling)。

用法: python test_json_schema.py
"""
import json
import httpx

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

# ── 测试 1: 检查模型是否在运行 ──
try:
    r = httpx.get("http://127.0.0.1:1234/v1/models", timeout=5)
    models = r.json()["data"]
    print("已加载模型:", [m["id"] for m in models if "qwen" in m["id"] or "gemma" in m["id"]])
except Exception as e:
    print(f"连接失败: {e}")
    exit(1)

# ── 测试 2: 带 response_format 的 Schema 约束请求 ──
schema_payload = {
    "model": "qwen/qwen3.5-9b",
    "messages": [
        {"role": "system", "content": "你是一个数据提取器，只输出 JSON。"},
        {"role": "user", "content": "列出 3 个国家的名称和首都。输出 JSON 对象。"}
    ],
    "temperature": 0,
    "max_tokens": 300,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "countries",
            "schema": {
                "type": "object",
                "properties": {
                    "countries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "capital": {"type": "string"}
                            },
                            "required": ["name", "capital"]
                        }
                    }
                },
                "required": ["countries"]
            }
        }
    }
}

print("\n[测试2] 发送 json_schema 约束请求...")
try:
    r = httpx.post(LMSTUDIO_URL, json=schema_payload, timeout=60)
    if r.status_code != 200:
        print(f"  状态 {r.status_code}")
        error = r.json() if r.text else {}
        msg = error.get("error", {}).get("message", r.text)
        print(f"  错误: {msg[:300]}")
        if "grammar" in str(msg).lower() or "schema" in str(msg).lower():
            print("  → 模型不支持 grammar/json_schema 约束")
        exit(1)

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"  成功! 输出: {content[:300]}")
    # Try parsing
    parsed = json.loads(content)
    print(f"  JSON 解析成功: countries={len(parsed.get('countries',[]))} 个国家")
    print("  ✅ LM Studio 支持 response_format json_schema!")
    print("  → 方案2 可以实施")

except Exception as e:
    print(f"  异常: {e}")

# ── 测试 3: 验证 schema 约束是否真的生效 ──
print("\n[测试3] 故意违反 schema（要求输出字段之外的数据）...")
schema_payload2 = {
    "model": "qwen/qwen3.5-9b",
    "messages": [
        {"role": "user", "content": "介绍中国和美国的首都，以及你自己对它们的看法。输出 JSON。"}
    ],
    "temperature": 0,
    "max_tokens": 300,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "countries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "countries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "capital": {"type": "string"}
                            },
                            "required": ["name", "capital"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["countries"],
                "additionalProperties": False
            }
        }
    }
}

try:
    r = httpx.post(LMSTUDIO_URL, json=schema_payload2, timeout=60)
    if r.status_code == 200:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        # Check if LLM tried to output extra fields (which should be blocked by strict schema)
        try:
            parsed = json.loads(content)
            countries = parsed.get("countries", [])
            extra_fields = any("opinion" in str(c).lower() or "看法" in str(c).lower() for c in countries)
            if extra_fields:
                print("  ⚠️ schema 未强制约束，LLM 仍输出了额外字段")
            else:
                print("  ✅ strict schema 生效，额外字段被禁止")
        except:
            print(f"  输出: {content[:200]}")
    else:
        error = r.json().get("error", {}).get("message", "") if r.text else ""
        print(f"  状态 {r.status_code}: {error[:200]}")
except Exception as e:
    print(f"  异常: {e}")
