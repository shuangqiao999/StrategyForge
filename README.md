# StrategyForge — 战略决策推演

**把战略决策变成可计算、可复现、可优化、可解释的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具：以一段「种子材料」为输入，自动构建知识图谱与智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"的闭环并行推演多轮，并能对多套策略做蒙特卡洛对比择优。它以「Python 后端 + Tauri 桌面应用」形态交付，可打包为独立安装包离线运行。

---

## 核心特性

- **五阶段推演流水线**：本体生成 → GraphRAG 知识图谱 → 智能体工厂 → 并行模拟 → 报告生成。
- **量化模式（规则包驱动）**：内置 5 个领域规则包（军事 / 商业 / 政治 / 生态 / 城市规划），用 `EntityState` + 规则效应做数值演化，由数值阈值与结构化胜利条件**客观判胜负**，根除"评估者悖论"。支持自定义规则包与自动领域识别；叙事模式（无规则包）由 LLM 评估。
- **多动作资源分配**（可选，会话级开关）：每个智能体每轮可把总预算按权重分配到多个动作、并可分别指向不同对手（混合策略 / 多线博弈），量级中性、默认关闭。
- **语义记忆与检索（LanceDB）**：原著切片的混合检索（向量 + 全文 FTS）+ 模拟事件的动态语义记忆 + 干预/目标显著性通道，并带查询/结果缓存。
- **知识图谱与因果链（Kuzu）**：实体与关系（RELATES）→ 关系反哺决策（注入盟友/对手、信任播种）；行动时序图（Event/ACTED）；**确定性因果链**（TARGETS/CAUSED，基于数值真值的精确归因），驱动"时间线 / 因果图"可视化与报告因果分析。
- **策略优化器**：对多套策略指令做 M×N 次隔离的蒙特卡洛推演，输出胜率/成本/帕累托前沿与推荐方案，并为推荐方案生成叙事报告、点亮时间线/因果页。
- **实时干预与预目标**：推演中注入用户指令；为会话设定不可变战略目标（pre-goal）。
- **桌面应用**：Tauri 2 壳（系统托盘 + 自动拉起后端），React 前端含 3D 力导图、报告、日志、时间线/因果图、优化对比等视图，内置 LLM/嵌入模型配置页。

---

## 技术架构

- **后端**：Python 3.11 + FastAPI/uvicorn（`strategy_forge.api:app`，默认 `http://127.0.0.1:8000`）。
- **桌面端**：Tauri 2（Rust 壳）+ React 18 + Vite + react-force-graph-3d / three。
- **数据存储**：
  - SQLite — 会话 / 日志 / 报告（`data/sessions.db`）
  - Kuzu — 知识图谱与时序因果（`data/graphs/{session}/kuzu`）
  - LanceDB — 向量/全文检索（`data/lancedb`）
  - `data/forge_config.json` — 端点与模型配置
- **LLM 接入**：OpenAI 兼容接口，统一 Provider 注册表，内置 OpenAI / DeepSeek / Kimi / 智谱 / 硅基流动 / 火山 / 通义 / Gemini / xAI / Groq / Ollama / LM Studio 等目录；对话与嵌入端点可分别配置。解析优先级：`forge_config.json` > `FORGE_*` 环境变量 > 厂商默认。

---

## 快速开始

### 1. 后端

```bash
# 安装（项目根目录）
pip install -e .

# 配置（任选其一）：复制 .env.example 为 .env 并编辑，或在桌面应用「配置」页设置
cp .env.example .env

# 启动开发服务器
python run.py
# 或：strategy-forge serve  (等价 python -m strategy_forge.main serve)
```

后端启动于 `http://127.0.0.1:8000`，文档 `http://127.0.0.1:8000/docs`，健康检查 `/health`。

### 2. 前端（桌面应用）

```bash
cd apps/strategy-forge
npm install
npm run dev        # 仅前端 (Vite, http://localhost:5173)
# 或
npx tauri dev      # 完整桌面应用（会自动构建并联调）
```

> 提示：推荐本地用 [LM Studio](https://lmstudio.ai) 或 Ollama 提供对话与嵌入模型（默认对话 `qwen/qwen3.5-9b`，嵌入 `text-embedding-embeddinggemma-300m-qat`）；也可在「配置」页填写任意云端 OpenAI 兼容服务商。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | 对话模型 API 地址 |
| `FORGE_LLM_KEY` | `lm-studio` | 对话模型 API Key |
| `FORGE_LLM_MODEL` | `qwen/qwen3.5-9b` | 对话模型 |
| `FORGE_EMBED_BASE` | 同 LLM | 嵌入模型 API 地址 |
| `FORGE_EMBED_KEY` | `lm-studio` | 嵌入模型 API Key |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | 嵌入模型 |
| `FORGE_PROVIDER` | （空） | 默认厂商标识（如 `lmstudio`/`openai`） |
| `FORGE_MAX_AGENTS` | `200` | 最大智能体数 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认模拟轮数 |
| `FORGE_CANDIDATE_COUNT` | `3` | 每轮候选策略数（定性模式） |
| `FORGE_LLM_TEMPERATURE` | `0.3` | 温度 |
| `FORGE_MAX_CONCURRENT` | `8` | 并发上限 |
| `FORGE_DATA_DIR` | `./data` | 数据目录 |

> 配置优先级：`forge_config.json`（前端/接口写入）> 上述 `FORGE_*` 环境变量 > 厂商默认。

---

## 使用流程

1. **新建会话**：填入标题与种子材料，选择推演领域（`auto` 自动识别 / 具体领域 / `narrative` 叙事 / `custom` 自定义规则包）。
2. **（可选）设定**：添加推演前目标（pre-goal）；在会话视图勾选「多动作」启用资源分配；用「干预」注入临时指令。
3. **运行**：
   - 「启动推演」：跑一次完整五阶段推演。
   - 「策略优化器」：对多套策略指令做蒙特卡洛对比，输出推荐方案。
4. **查看**：图谱（3D 关系网络）/ 报告（总结·风险·建议·量化终态·因果归因）/ 时间线·因果图（谁先做什么、对谁造成什么数值后果）/ 日志（实时 SSE）/ 优化（胜率·成本·帕累托·推荐）。

---

## 规则包（量化领域）

| 领域 | 关键指标 | 典型动作 |
|------|----------|----------|
| `military` 军事 | 兵力 / 士气 / 粮草 / 疲劳 / 指挥 | 进攻·防守·投资·机动·外交·观察 |
| `business` 商业 | 市场份额 / 现金流 / 品牌 / 研发 / 士气 | 价格战·研发·营销·扩张·结盟·观察 |
| `politics` 政治 | 支持率 / 立法权 / 国际关系 / 经济 / 团结 | 竞选·立法·外交·改革·攻击对手·观察 |
| `ecology` 生态 | 种群 / 资源 / 污染 / 多样性 / 稳定 | 开发·保护·治污·扩张·竞争·观察 |
| `urban` 城市规划 | 人口 / 就业 / 基建 / 财政 / 满意度 | 基建·产业·福利·引才·监管·观察 |

也可通过自定义规则包（领域 = `custom` + `custom_rules`）扩展任意领域。

---

## 打包与发布（Windows）

桌面安装包由 PyInstaller 后端 + Tauri NSIS 打包组成：

```bash
# 1. 打包后端（项目根目录，onedir）
python -m PyInstaller strategy-forge-backend.spec --noconfirm --clean

# 2. 拷入 Tauri 资源目录
#    dist/strategy-forge-backend/  ->  apps/strategy-forge/src-tauri/resources/strategy-forge-backend/

# 3. 构建桌面应用与安装包
cd apps/strategy-forge
npx tauri build
```

产物：
- 应用：`apps/strategy-forge/src-tauri/target/release/strategy-forge.exe`
- 安装包：`apps/strategy-forge/src-tauri/target/release/bundle/nsis/StrategyForge_0.1.0_x64-setup.exe`

运行时 Tauri 壳会自动以子进程拉起后端（`strategy-forge-backend.exe serve`）。

> 注意：`strategy-forge-backend.spec` 会将仓库 `data/` 快照打入安装包。对外分发前请清理/脱敏 `data/forge_config.json`（可能含 API Key）与历史会话库。

---

## 主要 API（前缀 `/api/forge`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传种子材料 |
| POST | `/session` | 创建会话 |
| GET | `/sessions` · `/session/{id}` | 列表 / 详情 |
| DELETE | `/session/{id}` | 删除会话（连带清理 Kuzu/LanceDB） |
| POST | `/session/{id}/start` | 启动五阶段推演 |
| POST | `/session/{id}/optimize` · `/optimize/cancel` · GET `/optimize/result` | 策略优化器 |
| POST | `/session/{id}/intervene` | 实时干预 |
| POST | `/session/{id}/pre-goal` | 设定推演前目标 |
| POST | `/session/{id}/settings` | 推演级设置（多动作） |
| GET | `/session/{id}/graph` · `/timeline` · `/causal` | 知识图谱 / 时间线 / 因果子图 |
| GET | `/session/{id}/report` · `/logs` · `/stream` | 报告 / 日志 / SSE 实时流 |
| GET/POST | `/api/forge/config/*` | LLM/嵌入端点与模型配置 |

---

## 项目结构

```
StrategyForge/
├── src/strategy_forge/
│   ├── core/        # 配置 / Provider 注册表 / LLM 适配 / 分词 / 规则包模板
│   ├── engine/      # 五阶段流水线 + 规则引擎 + 优化器 + 预处理器(LanceDB) + 推理器
│   ├── storage/     # SQLite 会话库 + Kuzu 图库
│   └── api/         # FastAPI 路由
├── apps/strategy-forge/        # Tauri 2 桌面应用 (React + 3D 图)
│   └── src-tauri/              # Rust 壳（拉起后端 + 托盘）
├── scripts/                    # 连接本地 LM Studio 的端到端测试脚本
├── data/                       # 运行期数据（SQLite / Kuzu / LanceDB / forge_config.json）
├── strategy-forge-backend.spec # PyInstaller 打包配置
├── release/                    # 打包产物（exe + 安装包）
├── run.py                      # 开发启动器
└── pyproject.toml
```

---

## 测试

`scripts/` 下提供连接本地 LM Studio（9B 对话 + embeddinggemma 嵌入）的端到端验证脚本：

```bash
python scripts/test_multi_action_lmstudio.py      # 多动作资源分配
python scripts/test_lancedb_fulllink_lmstudio.py  # LanceDB 语义记忆全链路
python scripts/test_graph_fulllink_lmstudio.py    # Kuzu 关系反哺 + 时序 + 因果（含确定性归因）
python scripts/test_optimizer_report_lmstudio.py  # 优化器叙事报告 + 多动作解耦
```

运行前请确保本地 LM Studio 已启动并加载对话与嵌入模型。

---

## 许可

AGPL-3.0-only
