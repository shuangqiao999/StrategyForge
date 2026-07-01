# StrategyForge — 战略决策推演

**把战略决策变成可计算、可复现、可优化、可解释的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具：以一段「种子材料」为输入，自动构建知识图谱与智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"的闭环并行推演多轮，并能对多套策略做蒙特卡洛对比择优。它以「Python 后端 + Tauri 桌面应用」形态交付，可打包为独立安装包离线运行。

---

## 核心特性

### 五阶段推演流水线

本体生成 → GraphRAG 知识图谱 → 智能体工厂 → 并行模拟 → 报告生成，全程自动化。

### 推演控制

- **启动 / 暂停 / 续推**：任意阶段可取消，进度自动持久化到 SQLite（`state_snapshot`）。退出应用后重新打开，从断点继续推演。
- **状态实时同步**：SSE 事件驱动，界面实时显示当前阶段（本体生成→图谱构建→智能体生成→模拟推演→报告生成）。
- **实时干预**：推演中注入用户指令，影响智能体决策方向。
- **不可变目标（pre-goal）**：为会话设定最终战略目标，贯穿全轮次。

### 模拟模式

- **量化模式（规则包驱动）**：内置 8 个领域规则包（军事 / 商业 / 政治 / 生态 / 城市 / 科技 / 信息战 / 地缘战略），用 `EntityState` + 规则效应做数值演化，由数值阈值与结构化胜利条件**客观判胜负**。支持自定义规则包与自动领域识别。
- **叙事模式**：无规则包时由 LLM 自由推理决策与判胜。
- **多动作资源分配**（会话级开关）：每轮可把总预算按权重分配到多个动作、并可分别指向不同对手（混合策略 / 多线博弈）。

### 算法模块（通用计算单元）

- **ODE 连续演化**（`algorithms/ode_module.py`）：N 实体 × M 指标的平滑数值变化。使用 scipy RK45 自适应积分（不可用时降级 numpy Euler 法）。内置 7 种预设方程（衰减/逻辑增长/疲劳恢复/供给消耗/污染扩散/资源消耗），可通过规则包 `modules` 段自定义。
- **3D 物理引擎**（`algorithms/physics_module.py`）：刚体动力学（欧拉积分）、AABB+球碰撞检测与响应、高斯扩散、径向爆炸冲击波。通过规则包 `modules` 段配置重力/阻尼/扩散率等参数。
- **规则包驱动配置**：规则包的 `modules` 段可指定每指标使用哪个 ODE 方程、物理子系统和参数。未定义时自动回退到预设匹配。

### 知识图谱与因果链（Kuzu）

实体与关系（RELATES）→ 关系反哺决策（注入盟友/对手、信任播种）；行动时序图（Event/ACTED）；**确定性因果链**（TARGETS/CAUSED，基于数值真值的精确归因），驱动"时间线 / 因果图"可视化与报告因果分析。

### 语义记忆与检索（LanceDB）

原著切片的混合检索（向量 + 全文 FTS）+ 模拟事件的动态语义记忆 + 干预/目标显著性通道，并带查询/结果缓存。

### 策略优化器

对多套策略指令做 M×N 次隔离的蒙特卡洛推演，输出胜率/成本/帕累托前沿与推荐方案，并为推荐方案生成叙事报告。

### Token 统计

每次 LLM 调用的输入/输出 token 数自动记录（通过 `contextvars` 无侵入捕获），按阶段/轮次汇总。前端提供汇总卡片 + SVG 柱状图 + 每 2 分钟自动刷新。

### 增强报告

- 量化轨迹注入 LLM Prompt（每轮指标变化值）
- 关键因果链卡片化 + 时序因果叙事阶段卡片
- 决策偏离分析（识别与不可变目标偏离的决策）
- 折叠式文档结构，支持快速浏览与深度阅读

### 桌面应用

Tauri 2 壳（系统托盘 + 自动拉起后端），React 前端含 3D 力导图、报告（折叠文档结构）、日志（实时 SSE）、Token 统计（柱状图）、时间线/因果图（节点点击文本联动）、优化对比等视图，内置 LLM/嵌入模型配置页。

---

## 技术架构

- **后端**：Python 3.11 + FastAPI/uvicorn（`strategy_forge.api:app`，默认 `http://127.0.0.1:8000`）。
- **桌面端**：Tauri 2（Rust 壳）+ React 18 + Vite + react-force-graph-3d / three。
- **数据存储**：
  - SQLite — 会话 / 日志 / 报告 / token 统计 / 暂停快照（`data/sessions.db`）
  - Kuzu — 知识图谱与时序因果（`data/graphs/{session}/kuzu`）
  - LanceDB — 向量/全文检索（`data/lancedb`）
  - `data/forge_config.json` — 端点与模型配置
- **LLM 接入**：OpenAI 兼容接口，统一 Provider 注册表，内置 28+ 厂商目录；对话与嵌入端点可分别配置。解析优先级：`forge_config.json` > `FORGE_*` 环境变量 > 厂商默认。
- **算法依赖**：`numpy>=1.24.0` + `scipy>=1.10.0`（可选，不可用时降级纯 numpy）。
- **许可证**：AGPL-3.0-only

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
2. **（可选）设定**：添加推演前目标（pre-goal）；在会话视图勾选「多动作」启用资源分配。
3. **运行**：
   - 「启动推演」：跑一次完整五阶段推演。运行中按钮变为绿底「推演中」，再次点击暂停并保存进度。
   - 「继续推演」：从上次暂停的轮次恢复，蓝底按钮。
   - 「策略优化器」：对多套策略指令做蒙特卡洛对比，输出推荐方案。
4. **查看**：
   - **图谱**：3D 关系网络（节点类型着色、关系标签）
   - **报告**：折叠文档结构，含最终状态、因果链卡片、时序叙事、偏离分析、结论建议
   - **时间线/因果图**：谁先做什么、对谁造成什么数值后果；点击节点查看关联文本
   - **日志**：实时 SSE 推送，所有阶段可见
   - **Token 统计**：汇总卡片 + SVG 柱状图（每轮输入/输出分色）
   - **优化**：胜率·成本·帕累托·推荐方案
5. **干预**：推演中在底部输入框发送指令，影响智能体决策。

---

## 规则包（量化领域）

内置 8 个领域规则包：

| 领域 | 关键指标 | 典型动作 |
|------|----------|----------|
| `military` 军事 | 兵力 / 士气 / 粮草 / 疲劳 / 指挥 | 进攻·防守·投资·机动·外交·围城·电子战·观察 |
| `business` 商业 | 市场份额 / 现金流 / 品牌 / 研发 / 士气 | 价格战·研发·营销·扩张·结盟·观察 |
| `politics` 政治 | 支持率 / 立法权 / 国际关系 / 经济 / 团结 | 竞选·立法·外交·改革·攻击对手·观察 |
| `ecology` 生态 | 种群 / 资源 / 污染 / 多样性 / 稳定 | 开发·保护·治污·扩张·竞争·观察 |
| `urban` 城市规划 | 人口 / 就业 / 基建 / 财政 / 满意度 | 基建·产业·福利·引才·监管·观察 |
| `tech` 科技 | 研发 / 专利 / 人才 / 融资 / 产品 | 研发·招聘·融资·发布·专利·观察 |
| `info_war` 信息战 | 舆论 / 渗透 / 防御 / 情报 / 士气 | 造谣·反制·渗透·情报·舆论·观察 |
| `geo_strategy` 地缘战略 | 12 指标（5 域融合） | 军事·经济·外交·文化·科技 各域动作 |

也可通过自定义规则包（领域 = `custom` + `custom_rules`）扩展任意领域。

### 规则包中的算法模块配置（可选）

在规则包 JSON 中添加 `modules` 段控制 ODE/Physics 模块行为：

```json
{
  "domain": "business",
  "metrics": ["market_share", "cash_flow", "brand", "rd_investment", "morale"],
  "modules": {
    "ode_engine": {
      "equations": {
        "market_share": "logistic",
        "cash_flow": "decay",
        "brand": "logistic"
      }
    },
    "physics_engine": {
      "gravity": 9.8,
      "damping": 0.95,
      "diffusion_rate": 0.1
    }
  }
}
```

`modules` 段可选。省略时自动按指标名称匹配内置预设（如 `fatigue`→疲劳恢复、`supply`→供给消耗）。

---

## 算法模块

`src/strategy_forge/algorithms/` 下的通用计算单元，领域无关、规则包驱动：

| 模块 | 文件 | 功能 |
|------|------|------|
| `ode_engine` | `ode_module.py` | 连续微分方程演化（scipy RK45 自适应积分） |
| `physics_engine` | `physics_module.py` | 3D 刚体动力学 / 碰撞 / 扩散 / 径向爆炸 |
| `base.py` | 基类 | `AlgorithmModule` + `ModuleContext` + `SpatialState` |
| `module_utils.py` | 工厂 | `build_module_chain()` 从规则包自动创建模块链 |

---

## 打包与发布（Windows）

桌面安装包由 PyInstaller 后端 + Tauri NSIS 打包组成：

```bash
# 1. 打包后端（项目根目录，onedir）
python -m PyInstaller strategy-forge-backend.spec --noconfirm

# 2. 拷入 data/rule/ 规则文件
#    data/rule/  →  dist/strategy-forge-backend/data/rule/

# 3. 拷入 Tauri 资源目录
#    dist/strategy-forge-backend/  →  apps/strategy-forge/src-tauri/resources/strategy-forge-backend/

# 4. 构建桌面应用与安装包
cd apps/strategy-forge
npx tauri build --bundles nsis
```

产物：
- 应用：`apps/strategy-forge/src-tauri/target/release/strategy-forge.exe`
- 安装包：`apps/strategy-forge/src-tauri/target/release/bundle/nsis/StrategyForge_0.1.0_x64-setup.exe`

运行时 Tauri 壳会自动以子进程拉起后端（`strategy-forge-backend.exe serve`）。

> 注意：`strategy-forge-backend.spec` 精确排除了 scipy 的重子模块（sparse/linalg/special 等），仅保留 `scipy.integrate`（RK45 自适应积分）。对外分发前请清理/脱敏 `data/forge_config.json`（可能含 API Key）与历史会话库。

---

## 主要 API（前缀 `/api/forge`）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传种子材料 |
| POST | `/session` | 创建会话 |
| GET | `/sessions` · `/session/{id}` | 列表 / 详情 |
| DELETE | `/session/{id}` | 删除会话（连带清理 Kuzu/LanceDB） |
| POST | `/session/{id}/start` | 启动五阶段推演 |
| POST | `/session/{id}/start/cancel` | 取消推演（暂停并保存进度） |
| POST | `/session/{id}/optimize` · `/optimize/cancel` · GET `/optimize/result` | 策略优化器 |
| POST | `/session/{id}/intervene` | 实时干预 |
| POST | `/session/{id}/pre-goal` | 设定推演前目标 |
| POST | `/session/{id}/settings` | 推演级设置（多动作等） |
| GET | `/session/{id}/graph` · `/timeline` · `/causal` | 知识图谱 / 时间线 / 因果子图 |
| GET | `/session/{id}/report` · `/logs` · `/stream` | 报告 / 日志 / SSE 实时流 |
| GET | `/session/{id}/tokens` | Token 统计 |
| GET/POST | `/config/*` | LLM/嵌入端点与模型配置 |
| GET | `/domains` | 可用规则包领域列表 |
| POST | `/rules/upload` | 上传自定义规则包 |

---

## 项目结构

```
StrategyForge/
├── src/strategy_forge/
│   ├── algorithms/     # 通用算法模块（ODE / Physics / 基类）
│   ├── core/           # 配置 / Provider 注册表 / LLM 适配 / token 统计
│   ├── engine/         # 五阶段流水线 + 规则引擎 + 优化器 + 预处理器(LanceDB) + 推理器
│   ├── storage/        # SQLite 会话库 + Kuzu 图库
│   └── api/            # FastAPI 路由 + SSE 事件流
├── apps/strategy-forge/          # Tauri 2 桌面应用 (React + 3D 图)
│   └── src-tauri/                # Rust 壳（拉起后端 + 托盘）
├── scripts/                      # 端到端测试脚本（LM Studio）
├── tests/                        # 单元测试 + 功能测试
├── data/                         # 运行期数据（SQLite / Kuzu / LanceDB / forge_config.json）
│   └── rule/                     # 内置规则包 + 自定义规则包目录
├── strategy-forge-backend.spec   # PyInstaller 打包配置
├── release/                      # 打包产物（exe + 安装包）
├── run.py                        # 开发启动器
└── pyproject.toml
```

---

## 测试

### 单元测试

```bash
# 图连接竞态 + 算法模块
python -m pytest tests/ -v
```

### 端到端测试（需 LM Studio）

`scripts/` 下提供连接本地 LM Studio 的验证脚本：

```bash
python scripts/test_algorithms_lmstudio.py        # ODE + Physics + Token + 全流程（5 项）
python scripts/test_quantified_lmstudio.py        # 量化推演闭环
python scripts/test_multi_action_lmstudio.py      # 多动作资源分配
python scripts/test_graph_fulllink_lmstudio.py    # Kuzu 关系反哺 + 时序 + 因果
python scripts/test_optimizer_lmstudio.py         # 蒙特卡洛优化器
```

运行前请确保本地 LM Studio 已启动并加载对话与嵌入模型。
