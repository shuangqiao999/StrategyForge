# StrategyForge — 战略决策推演

**把战略决策变成可计算、可复现、可优化的科学实验。**

StrategyForge 是从 OpenAkita 推演引擎剥离出来的专职战略决策推演工具。它保留完整的五阶段推演流水线（本体生成 → GraphRAG 知识图谱 → 智能体工厂 → 并行模拟 → 报告生成），以独立工具形态存在。

---

## 快速启动

### 1. 安装依赖

```bash
cd backend
pip install -e .
```

### 2. 配置 LLM

复制 `.env.example` 为 `.env` 并编辑：

```bash
cp .env.example .env
```

### 3. 启动后端

```bash
python run.py
```

后端启动在 `http://127.0.0.1:8000`，API 文档在 `http://127.0.0.1:8000/docs`。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端启动在 `http://localhost:5173`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | LLM API 地址 |
| `FORGE_LLM_MODEL` | `qwen/qwen3.5-9b` | 对话模型 |
| `FORGE_EMBED_BASE` | 同 LLM | 嵌入 API 地址 |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | 嵌入模型 |
| `FORGE_MAX_AGENTS` | `200` | 最大智能体数 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认模拟轮数 |
| `FORGE_CANDIDATE_COUNT` | `3` | 每轮候选策略数 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/forge/session` | 创建推演会话 |
| GET | `/api/forge/sessions` | 会话列表 |
| GET | `/api/forge/session/{id}` | 会话详情 |
| POST | `/api/forge/session/{id}/start` | 启动推演 |
| POST | `/api/forge/session/{id}/intervene` | 实时干预 |
| POST | `/api/forge/session/{id}/pre-goal` | 设定预目标 |
| GET | `/api/forge/session/{id}/graph` | 图谱数据 |
| GET | `/api/forge/session/{id}/report` | 推演报告 |
| GET | `/api/forge/session/{id}/logs` | 日志 |
| GET | `/api/forge/session/{id}/stream` | SSE 实时流 |
| DELETE | `/api/forge/session/{id}` | 删除会话 |
| POST | `/api/forge/upload` | 上传文档 |

---

## 项目结构

```
backend/
├── src/
│   ├── core/          # 配置 + LLM 适配器 + 分词器
│   ├── engine/        # 五阶段推演流水线
│   ├── storage/       # Kuzu 图数据库 + SQLite 会话
│   └── api/           # FastAPI 路由
frontend/
└── src/
    ├── App.tsx        # 唯一主视图
    └── types/         # TypeScript 类型
```

---

## 许可

AGPL-3.0-only
