# StrategyForge — 战略决策推演

**把战略决策变成可计算、可复现、可优化、可解释的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具：以一段「种子材料」为输入，自动构建知识图谱与智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"的闭环并行推演多轮，并能对多套策略做蒙特卡洛对比择优。它以「Python 后端 + Tauri 桌面应用」形态交付，可打包为独立安装包离线运行。

---

## 核心特性

### 五阶段推演流水线

本体生成 → GraphRAG 知识图谱 → 智能体工厂 → 并行模拟 → 报告生成，全程自动化。

### 推演控制

- **启动 / 暂停 / 继续**：任意轮次可暂停，当前轮正在执行的 LLM 调用完成后立即中断，进度自动持久化到 SQLite（`state_snapshot`）。退出应用后重新打开，从断点继续推演。
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

Tauri 2 壳（系统托盘 + 自动拉起后端），React 18 前端含 3D 力导图、报告（折叠文档结构）、日志（实时 SSE）、Token 统计（柱状图）、时间线/因果图（节点点击文本联动）、优化对比等视图，内置 LLM/嵌入模型配置页。

---

## 技术架构

- **后端**：Python 3.11 + FastAPI/uvicorn（`strategy_forge.api:app`，默认 `http://127.0.0.1:8000`）。
- **桌面端**：Tauri 2（Rust 壳，系统托盘 + 自动拉起/关闭后端进程）+ React 18 + TypeScript + Vite + react-force-graph-3d / three.js。
- **数据存储**：
  - SQLite — 会话 / 日志 / 报告 / token 统计 / 暂停快照（`data/sessions.db`）
  - Kuzu — 知识图谱与时序因果（`data/graphs/{session}/kuzu`）
  - LanceDB — 向量/全文检索（`data/lancedb`）
  - `data/forge_config.json` — 端点与模型配置
- **LLM 接入**：OpenAI 兼容接口，统一 Provider 注册表，内置 28+ 厂商目录；对话与嵌入端点可分别配置。解析优先级：`forge_config.json` > `FORGE_*` 环境变量 > 厂商默认。
- **算法依赖**：`numpy>=1.24.0` + `scipy>=1.10.0`（可选，不可用时降级纯 numpy Euler 法）。
- **许可证**：AGPL-3.0-only

---

## 快速开始

### 1. 后端

```bash
# 安装（项目根目录）
pip install -e .

# 配置：复制 .env.example 为 .env 并编辑（或在桌面应用「配置」页设置）
cp .env.example .env

# 启动开发服务器
python run.py
# 或：strategy-forge serve
```

后端启动于 `http://127.0.0.1:8000`，文档 `http://127.0.0.1:8000/docs`，健康检查 `/health`。

### 2. 前端（桌面应用）

```bash
cd apps/strategy-forge
npm install
npm run dev        # 仅前端 (Vite, http://localhost:5173)
# 或
npx tauri dev      # 完整桌面应用（自动构建并联调）
```

> 推荐本地用 [LM Studio](https://lmstudio.ai) 或 Ollama 提供对话与嵌入模型（默认对话 `qwen/qwen3.5-9b`，嵌入 `text-embedding-embeddinggemma-300m-qat`）；也可在「配置」页填写任意云端 OpenAI 兼容服务商。

---

## 环境变量

所有带 `FORGE_` 前缀的环境变量均有 `data/forge_config.json` 覆盖机制。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | 对话模型 API 地址 |
| `FORGE_LLM_KEY` | `lm-studio` | 对话模型 API Key |
| `FORGE_LLM_MODEL` | `qwen/qwen3.5-9b` | 对话模型 ID |
| `FORGE_EMBED_BASE` | 同 LLM | 嵌入模型 API 地址 |
| `FORGE_EMBED_KEY` | `lm-studio` | 嵌入模型 API Key |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | 嵌入模型 ID |
| `FORGE_PROVIDER` | （空） | 默认厂商标识（如 `lmstudio` / `openai`） |
| `FORGE_MAX_AGENTS` | `10000` | 最大智能体数 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认模拟轮数 |
| `FORGE_CANDIDATE_COUNT` | `3` | 每轮候选策略数（定性模式） |
| `FORGE_LLM_TEMPERATURE` | `0.3` | LLM 温度 |
| `FORGE_MAX_CONCURRENT` | `2` | 并发 LLM 请求上限 |
| `FORGE_RETRIEVE_TOP_K` | `5` | LanceDB 检索返回 Top-K |
| `FORGE_SIMILARITY_THRESHOLD` | `0.4` | 语义检索相似度阈值 |
| `FORGE_DATA_DIR` | `./data` | 运行期数据目录 |
| `FORGE_RULE_DIR` | （空） | 内置规则包目录（桌面打包时由 Tauri 壳设置） |

> 以上默认值为 `config.py` 中 `DeductionConfig` 的硬编码值；`.env.example` 仅作示例，实际以代码为准。

---

## 使用流程

1. **新建会话**：填入标题与种子材料，选择推演领域（`auto` 自动识别 / 具体领域 / `narrative` 叙事 / `custom` 自定义规则包）。
2. **（可选）设定**：添加推演前目标（pre-goal）；在会话视图勾选「多动作」启用资源分配。
3. **运行**：
   - **「启动推演」**（蓝色按钮）：跑一次完整五阶段推演。运行中按钮变为**红色「停止推演」**，左侧有绿色"推演中"标签。
   - **「继续推演」**（绿色按钮）：从上次暂停的轮次恢复，载入快照跳过已完成的阶段。
   - **「策略优化器」**：对多套策略指令做蒙特卡洛对比，输出推荐方案。
4. **查看**：
   - **图谱**：3D 力导向图（节点按实体类型着色、关系标签悬停显示）
   - **报告**：折叠文档结构，含最终状态、因果链卡片、时序叙事、偏离分析、结论建议
   - **时间线/因果图**：谁先做什么、对谁造成什么数值后果；点击节点查看关联文本
   - **日志**：实时 SSE 推送，五阶段全部可见
   - **Token 统计**：汇总卡片 + SVG 柱状图（每轮输入/输出分色）
   - **优化**：胜率·成本·帕累托·推荐方案
5. **干预**：推演中在底部输入框发送指令，影响智能体决策方向。

---

## 策略优化器

策略优化器是推演引擎上层的蒙特卡洛对比层：对同一份种子材料，按多套不同"战略指令"分别跑 N 次隔离推演，由 LLM 依据统一"胜利条件"评估每次结局，输出统计对比结果和推荐方案。

### 适用场景

| 场景 | 举例 |
|------|------|
| 策略 A/B 对比 | "坚决北伐" vs "先平定南方再北伐" |
| 关键决策分歧 | "是否接受招安"、"是否出兵援救盟友" |
| 参数调优 | 量化模式下同一策略不同权重分配（如进攻/防守资源比） |
| 蒙特卡洛稳健性 | 同一策略反复运行 N 次，评估外部随机性对结局的影响 |

### 核心概念

- **方案（Scenario）**：一套战略指令文本。优化器按方案分组统计。
- **胜利条件（Win Condition）**：统一判定所有方案的准绳。留空则自动使用会话的"推演前目标（pre-goal）"。
- **隔离运行**：每次模拟使用独立内存（`persist_events=False`），M×N 次推演互不污染。
- **优化目标**：
  - 📈 最高胜率 — 选胜率最高的方案
  - 💰 最低成本/风险 — 选成本最低的方案
  - ⚖️ 平衡（帕累托最优）— 在胜率与成本的二维空间中筛选非劣解集

### 使用步骤

1. **勾选左侧栏「策略优化器」开关**，展开配置面板。
2. **设置模拟次数**：滑块 2–100 次/方案。总推演次数 = 方案数 × 次数，建议先用小次数（如 10）快速试跑。
3. **填写胜利条件**（非必填）：
   - 量化模式：如 `我方实体存活且敌军被消灭`，LLM 按数值阈值客观判胜。
   - 叙事模式：LLM 自由理解条件文本，注意语言精确性。
4. **添加方案**：
   - 方案名：简短标签，如"激进进攻方案"
   - 战略指令：一段自然语言描述，<100 字为宜，如 `集中兵力闪电突袭君士坦丁堡，不外交不招安`
   - 我方实体：量化模式下指定判胜指标绑定的实体（如 `甲军团`），留空 = 全体存活率
5. **点击「启动优化」**，等待 2 秒轮询的进度条更新。
6. **查看结果**：
   - 进度条 + 当前方案/当前最高胜分实时更新
   - 结果卡片：推荐方案高亮，显示胜率均值 + 95% 置信区间 + 成功率 + 成本均值
   - 胜率/成本散点图：每个方案一个点，帕累托前沿以绿色高亮
   - 各方案统计表：每方案的完整统计信息 + 策略指令回显

### 结果解读

| 指标 | 含义 |
|------|------|
| 胜率（Win Mean） | 胜利程度的均值（0–1），越高越好 |
| 成功率（Success Rate） | 达成胜利条件的比例（0–1） |
| 成本（Cost Mean） | 付出的代价/风险均值（0–1），越低越好 |
| 95% CI | 胜率的 95% 置信区间，反映样本波动 |
| 帕累托前沿 | 在"高胜率"与"低成本"两维度上不被其他任何方案同时超越的方案集 |

### 技术细节

- 优化器自己运行一次 Phase 1–3（本体→图谱→智能体），M 个方案共享该基线。
- 每次模拟前深拷贝基线状态，用不同随机种子和微小温度扰动（±0.1）保证多样性。
- 每轮决策由 `StrategicReasoner` 基于用户指令 + 当前世界状态做策略对齐评分。
- 量化模式下，判胜由 `RuleEngine.judge()` 按数值阈值客观评定；叙事模式下由 LLM 依据胜利条件做主观评估。

---

## 规则包（量化领域）

规则包是量化推演的核心驱动组件：它将离散的 LLM 决策意图映射为结构化的数值效应，通过指标演化、阈值存亡判定、结构化胜利条件实现**客观判胜负**。规则包以 JSON 文件存储，内置 8 个领域，支持用户自定义扩展。

### 规则包结构

一个完整的规则包 JSON 对象包含以下字段：

```json
{
  "domain": "military",
  "name": "⚔️ 军事战争",
  "display_name": "军事战争",
  "metrics": ["strength", "morale", "supply", "fatigue", "leadership"],
  "initial_metrics": {"strength": 100, "morale": 75, "supply": 85, "fatigue": 10, "leadership": 70},
  "thresholds": {"strength": 15, "morale": 20, "supply": 10, "leadership": 15},
  "actions": ["attack", "defend", "invest", "maneuver", "diplomacy", "observe"],
  "self_effects": { ... },
  "target_effects": { ... },
  "conditional_effects": { ... },
  "delay_effects": { ... },
  "auto_effects": { ... },
  "weather_modifiers": { ... },
  "terrain_modifiers": { ... },
  "modules": { ... }
}
```

#### 顶层字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|:---:|------|
| `domain` | string | ✓ | 领域唯一标识（如 `military`），同领域名覆盖加载 |
| `name` / `display_name` | string | — | UI 显示名称 |
| `metrics` | list[string] | ✓ | 量化指标列表，如 `["strength", "morale", "supply"]` |
| `initial_metrics` | dict | ✓ | 每个指标的初始值（0–100 或自定义量程） |
| `thresholds` | dict | ✓ | 存亡阈值：任一指标 *严格低于* 阈值则实体出局 |
| `actions` | list[string] | — | 可选动作列表，供 LLM 决策时选择（默认 `["observe"]`） |
| `metric_ranges` | dict | — | 指标的合法范围，如 `{"strength": [0, 100]}`。缺省为 `[0, 100]` |
| `weather_modifiers` | dict | — | 天气修饰（可选 key: `rain`/`snow`/`clear`），环境效应附加到自身 |
| `terrain_modifiers` | dict | — | 地形修饰（可选 key: `mountain`/`plain`/`forest`），同天气 |

### 效应系统

规则包的效应层分为五个维度，按以下优先级依次计算：

#### 1. 自身效应（self_effects）

动作对**执行者自身**的直接数值影响。每个 key 是动作名，value 是 `{指标: 变化量}`。

```json
"self_effects": {
  "attack":  {"strength": -15, "supply": -12, "fatigue": 10, "morale": -5},
  "defend":  {"strength": -4, "morale": 2, "fatigue": 3, "supply": -3},
  "observe": {"fatigue": -5, "morale": 1}
}
```

> 所有效应值在实际结算时乘以 LLM 输出的 `intensity`（0–1），表示动作的执行力度。

#### 2. 目标效应（target_effects）

动作对**目标实体**的数值影响，结构与 `self_effects` 相同。仅在 LLM 指定了有效 `target` 时结算。

```json
"target_effects": {
  "attack": {"strength": -20, "morale": -12, "supply": -8},
  "diplomacy": {"morale": 3}
}
```

#### 3. 条件效应（conditional_effects）

动作在特定**自身状态条件**满足时触发的额外效应。key 格式为 `{action}_{condition_name}`，`condition` 字段支持简单布尔表达式。

```json
"conditional_effects": {
  "attack_morale_low": {
    "condition": "morale < 30",
    "self_effects": {"strength": -25, "fatigue": 5}
  },
  "blitzkrieg_fatigue": {
    "condition": "strength > 80 and fatigue < 20",
    "self_effects": {"fatigue": 5}
  }
}
```

> `condition` 语法：`指标名 运算符 数值`，支持 `and` / `or` 连接，运算符：`<` `>` `<=` `>=` `==` `!=`。系统预先编译为结构化形式，避免每轮字符串解析。

#### 4. 延迟效应（delay_effects）

动作的一部分效果延期结算（如投资需要 2 轮才能见效）。

```json
"delay_effects": {
  "invest": {"delay": 2, "effects": {"strength": 8, "morale": 3, "leadership": 2}}
}
```

| 字段 | 说明 |
|------|------|
| `delay` | 延迟轮数（≥1），在第 `current_round + delay` 轮结算 |
| `effects` | 延期结算的效应值（不受 `intensity` 缩放） |

#### 5. 自动效应（auto_effects）

**每轮开始前**自动检测并应用的效应，不依赖用户决策。用于模拟系统级变化（如疲劳恢复、供给消耗、士气衰退）。

```json
"auto_effects": {
  "fatigue_recovery": {
    "condition": "fatigue > 20",
    "effects": {"fatigue": -4}
  },
  "supply_drain": {
    "condition": "strength > 50",
    "effects": {"supply": -3}
  },
  "attrition": {
    "condition": "supply < 20",
    "effects": {"strength": -5, "morale": -10, "leadership": -5}
  },
  "logistics_snarl": {
    "condition": "supply < 30 and fatigue > 50",
    "effects": {"strength": -3, "morale": -3}
  }
}
```

### 胜利条件与胜负判定

`RuleEngine.judge()` 提供双轨判胜逻辑：

#### 量化判胜（有 win_target）

当策略优化器或会话配置指定了 `win_target`（绑定指标阈值）时：

```json
{
  "entity_ref": "甲军团",
  "metrics": {"strength": 30, "morale": 20},
  "threshold_logic": "all"
}
```

| `threshold_logic` | 含义 |
|:---|------|
| `all`（默认） | 所有指定指标都达到阈值才算成功 |
| `any` | 任一指标达到阈值即算成功 |
| `weighted_score` | 达到阈值程度的加权均值 ≥ 0.5 |

#### 默认判胜（无 win_target）

按所有阈值约束指标相对初值的损耗均值计算 `win_score`；存亡判定（`is_alive`）决定是否成功。

`cost` 值统一计算：阈值约束指标相对初始值的损耗均值（0–1，越低越好）。

### 内置领域

| 领域 | 指标数 | 动作数 | 说明 |
|------|:---:|:---:|------|
| `military` 军事战争 | 5 | 8 | 进攻·防守·投资·机动·外交·围城·电子战·观察 |
| `business` 商业竞争 | 6 | 10 | 价格战·研发·营销·扩张·合作·上市·禁运·挖角·观察 |
| `politics` 政治博弈 | 6 | 7 | 竞选·立法·外交·改革·攻击对手·观察 |
| `ecology` 生态模拟 | 7 | 7 | 开发·保护·治污·扩张·竞争·观察 |
| `urban` 城市规划 | 5 | 6 | 基建·产业·福利·引才·监管·观察 |
| `tech` 科技创新 | 5 | 7 | 研发·招聘·融资·发布·专利·观察 |
| `info_war` 信息战 | 5 | 6 | 造谣·反制·渗透·情报·舆论·观察 |
| `geo_strategy` 地缘战略 | 12 | 6 | 跨军事/经济/外交/文化/科技五域，含天气/地形/ODE 联动 |

### 自定义规则包

#### 方式一：桌面应用上传

在「策略优化器」面板中，选择领域 = `custom`，点击上传规则包 JSON 文件（`POST /api/forge/rules/upload`）。

#### 方式二：直接放置文件

将 JSON 文件放入 `data/rule/custom/` 目录，后端调用 `reload_rules()` 或重启后生效。

#### 方式三：开发模式

直接编辑 `data/rule/rules.json` 或新建自定义 JSON 文件。

#### 加载优先级

1. `FORGE_RULE_DIR` 目录（桌面打包时安装目录）→ 内置规则（只读）
2. `data/rule/custom/` → 用户自定义规则（持久化）
3. 同名领域：后加载的覆盖先加载的（自定义优先）
4. 无文件时：回退到代码内置的 5 个基础领域

### 规则包中的算法模块配置

`modules` 段为可选的算法模块驱动配置：

```json
"modules": {
  "metric_ownership": {
    "_doc": "指定哪些指标由 ODE 模块处理，其余由 rule_engine 管理",
    "fatigue": "ode",
    "supply": "ode",
    "cash_flow": "ode",
    "tech_lead": "ode"
  },
  "ode_engine": {
    "equations": {
      "fatigue": "fatigue_recovery",
      "supply": "supply_consumption",
      "cash_flow": "decay",
      "tech_lead": "logistic"
    }
  },
  "physics_engine": {
    "gravity": 9.8,
    "damping": 0.95,
    "collision_elasticity": 0.3,
    "diffusion_rate": 0.1
  }
}
```

| 子段 | 说明 |
|------|------|
| `metric_ownership` | 声明每个指标的计算归属（`"ode"` / `"physics"` / 省略=rule_engine） |
| `ode_engine` | ODE 方程分配：每指标 → 预设方程名（`fatigue_recovery` / `supply_consumption` / `decay` / `logistic` / `growth` / `pollution` / `resource_depletion`） |
| `physics_engine` | 3D 物理引擎参数：重力/阻尼/碰撞弹性/扩散率/爆炸力等 |

`modules` 段完全可选。省略时自动按指标名称匹配内置预设（如 `fatigue`→疲劳恢复、`supply`→供给消耗）。

### 应用场景

| 场景 | 推荐领域 | 说明 |
|------|----------|------|
| 战役推演 | `military` | 五维军事指标，8 种战法，含围城/电子战/饿死条件 |
| 商业博弈 | `business` | 市场份额·现金流·品牌·研发，支持上市/禁运/挖角 |
| 选举/政策模拟 | `politics` | 支持率·立法权·国际关系·经济·团结 |
| 环境保护 | `ecology` | 种群·资源·污染·多样性平衡模拟 |
| 城市规划 | `urban` | 人口·就业·基建·财政·满意度 |
| 科技竞赛 | `tech` | 研发·专利·人才·融资·产品生命周期 |
| 信息战 | `info_war` | 舆论·渗透·防御·情报·士气 |
| 大国博弈 | `geo_strategy` | 12 指标五域融合，跨军事/经济/外交/文化/科技 |
| 自定义逻辑 | `custom` | 上传自定义 JSON，定义任意领域和指标系统 |

### 编写新规则包

```json
{
  "domain": "zombie_apocalypse",
  "name": "🧟 僵尸末日",
  "display_name": "僵尸末日",
  "metrics": ["population", "food", "ammo", "morale", "infection_rate"],
  "initial_metrics": {
    "population": 80, "food": 60, "ammo": 50, "morale": 40, "infection_rate": 5
  },
  "thresholds": {"population": 10, "food": 5, "morale": 10},
  "actions": ["scavenge", "fortify", "rescue", "attack_horde", "trade", "observe"],
  "self_effects": {
    "scavenge":     {"ammo": 5, "food": 8, "morale": -3, "infection_rate": 5},
    "fortify":      {"ammo": -5, "food": -3, "morale": 5},
    "rescue":       {"population": 5, "food": -10, "morale": 8, "infection_rate": 3},
    "attack_horde": {"ammo": -15, "food": -3, "morale": 3, "population": -3},
    "trade":        {"food": 5, "ammo": 5, "morale": 2},
    "observe":      {"morale": 1}
  },
  "target_effects": {
    "attack_horde": {"population": -10, "morale": -5},
    "trade":        {"food": -3, "ammo": -3}
  },
  "conditional_effects": {
    "scavenge_dangerous": {
      "condition": "infection_rate > 30",
      "self_effects": {"population": -5, "morale": -5}
    },
    "fortify_desperate": {
      "condition": "population < 30",
      "self_effects": {"morale": 10}
    }
  },
  "delay_effects": {
    "fortify": {"delay": 2, "effects": {"population": 2, "morale": 5}}
  },
  "auto_effects": {
    "starvation": {
      "condition": "food < 15",
      "effects": {"population": -3, "morale": -8, "infection_rate": 3}
    },
    "infection_spread": {
      "condition": "infection_rate > 40",
      "effects": {"population": -5, "morale": -5}
    },
    "natural_recovery": {
      "condition": "infection_rate < 20 and food > 40",
      "effects": {"population": 1, "infection_rate": -2}
    }
  },
  "weather_modifiers": {
    "snow":  {"food": -10, "morale": -5, "ammo": -5},
    "clear": {"morale": 2}
  }
}
```

> 以下是编写新规则包的核心原则：
> - **指标量程尽量统一**（建议 0–100），避免不同指标的效应尺度差异过大。
> - **自身效应反映执行成本**（消耗供给、增加疲劳），目标效应反映对敌影响。
> - **至少定义 3–5 个有意义的阈值**，确保 LLM 能从数值变化中感知策略优劣。
> - **自动效应用于模拟环境压力**（如食物短缺导致人口/士气下降），让推演不静止。
> - **延迟效应用于建模投资/建设类动作**，避免"当轮决定当轮见效"的即时满足偏差。
> - **条件效应用于动态环境下的差异化反馈**，避免所有状态下相同动作恒定效应。

---

## 算法模块

`src/strategy_forge/algorithms/` 下的领域无关通用计算单元，由规则包驱动，通过 `ModuleContext` 与 `SpatialState` 做纯数据交换，不依赖具体领域知识。

### 架构概述

```
规则包 (rules.json modules 段)
    │
    ▼
module_utils.build_module_chain()    ── 工厂：创建 ODE + Physics 模块链
    │
    ├─ ODEModule.execute(ctx)        ── 连续数值演化
    └─ PhysicsModule.execute(ctx)    ── 空间物理模拟
    │
    ▼
module_utils.apply_context_results() ── 模块输出写回 EntityState
```

每轮推演的算法模块调用流程：
1. `build_context()` — 从 `EntityState` 字典构建 `ModuleContext`（含 numpy 数组形式的指标 + `SpatialState` 空间状态）
2. 顺序执行模块链（ODE → Physics），各模块原地修改 `ctx`
3. `apply_context_results()` — 将 `ctx.arrays` 写回各实体的 `EntityState.metrics`

---

### ODE 模块（ode_engine）

文件：`algorithms/ode_module.py`

**功能**：N 实体 × M 指标的连续微分方程演化。将离散的动作效应平滑到连续的 dt 时间段内，使指标变化符合物理/经济规则的渐进性，而非跳跃式突变。

**数学方法**：
- 有 scipy：`scipy.integrate.solve_ivp` 的 RK45 自适应步长积分（`rtol=1e-3, atol=1e-4`）
- 无 scipy：降级为 numpy Euler 法（`sub_steps` 控制子步数）

**内置预设方程**：

| 预设名 | 数学形式 | 适用指标 |
|--------|----------|----------|
| `decay` | dy/dt = −0.02 × y | 现金、资源自然衰减 |
| `logistic` | dy/dt = 0.03 × y × (1 − y/100) | 人口、市场份额、品牌 |
| `fatigue_recovery` | dy/dt = −0.05 × √y | 疲劳度恢复 |
| `supply_consumption` | dy/dt = −0.3 − 0.01 × |strength|/100 | 供给持续消耗 |
| `pollution_spread` | dy/dt = 0.001×factories − 0.05×greens − 0.01×y | 环境污染扩散 |
| `resource_depletion` | dy/dt = −0.005 × |population| | 资源随人口消耗 |

**跨指标依赖**：`supply_consumption` 依赖 `strength`；`pollution_spread` 依赖 `factory_output` 和 `green_coverage`；`resource_depletion` 依赖 `population`。缺失时模块会发出警告。

**规则包配置示例**：

```json
"modules": {
  "metric_ownership": {
    "fatigue": "ode",
    "supply": "ode",
    "cash_flow": "ode"
  },
  "ode_engine": {
    "sub_steps": 8,
    "equations": {
      "fatigue": "fatigue_recovery",
      "supply": "supply_consumption",
      "cash_flow": "decay",
      "market_share": "logistic"
    }
  }
}
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `equations` | 自动按指标名匹配预设 | `{指标名: 预设方程名}` 映射 |
| `sub_steps` | 4 | Euler 降级时的子步数；RK45 时无效 |

**应用场景**：

| 场景 | 配置 |
|------|------|
| 疲劳度自然恢复 | `fatigue: fatigue_recovery` |
| 供给随兵力消耗 | `supply: supply_consumption` |
| 现金随时间衰减 | `cash_flow: decay` |
| 市场份额增长 | `market_share: logistic` |
| 污染扩散 | `pollution: pollution_spread` |

---

### 物理模块（physics_engine）

文件：`algorithms/physics_module.py`

**功能**：3D 空间物理模拟，包含四个可选子系统，按顺序执行：

```
动力学 (Euler 积分) → 碰撞检测与响应 → 高斯扩散 → 径向爆炸冲击波
```

#### 子系统 1：刚体动力学（dynamics）

- 欧拉积分：`acceleration = forces / mass`（含重力 −9.8 m/s² 沿 Z 轴）
- 速度阻尼：`velocity *= damping`（每帧）
- dt 钳制：`dt = min(dt, 0.5)` 防止大时间步不稳定
- 每帧清空受力池（`forces.fill(0.0)`）

#### 子系统 2：碰撞检测与响应（collision）

- **实体数 ≤ 150**：暴力 O(N²) 逐对检测
- **实体数 > 150**：空间哈希（cell size = 2 × max_radius），3×3×3 邻域查询，趋近 O(N)
- 碰撞响应：完全弹性碰撞公式（质量加权速度交换）+ 分离（推至 min_dist 距离）

#### 子系统 3：高斯扩散（diffusion）

- 对 `diffusion_fields` 中指定的指标做空间上相邻实体之间的数值扩散
- 权重：`exp(−distance² / σ²)`，σ = radius × sigma_scale
- 速率因子 `diffusion_rate` 控制每轮扩散强度
- 用于模拟：技术溢出、经济辐射、污染扩散

#### 子系统 4：径向爆炸（explosion）

- 静态源：规则包 `explosion_sources` 段配置
- 动态触发：`ctx.metadata["trigger_explosion"]` 列表（运行时可注入）
- 每个爆炸源施加径向力：`force = power × (1 − distance/radius)`，衰减到 radius 外为零
- 力方向：从爆炸中心指向实体

**规则包配置示例**：

```json
"modules": {
  "physics_engine": {
    "subsystems": ["dynamics", "collision", "diffusion", "explosion"],
    "gravity": 9.8,
    "damping": 0.95,
    "collision_elasticity": 0.3,
    "diffusion_rate": 0.1,
    "diffusion_sigma_scale": 3.0,
    "diffusion_fields": ["technology", "economy"],
    "explosion_sources": [
      {"center": [0, 0, 0], "power": 50.0, "radius": 30.0}
    ]
  }
}
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `subsystems` | 全部启用 | 可关闭不需要的子系统以提效 |
| `gravity` | 9.8 | 重力加速度（m/s²） |
| `damping` | 0.98 | 速度阻尼系数（0–1，越接近 1 衰减越慢） |
| `collision_elasticity` | 0.5 | 碰撞弹性（0=完全非弹性，1=完全弹性） |
| `diffusion_rate` | 0.05 | 扩散速率 |
| `diffusion_sigma_scale` | 3.0 | 扩散 σ 缩放（相对实体半径） |
| `diffusion_fields` | `[]` | 需要空间扩散的指标名列表 |
| `explosion_sources` | `[]` | 静态爆炸源列表 |

**应用场景**：

| 场景 | 配置要点 |
|------|----------|
| 军事冲突 | 开启 collision + explosion；爆炸源标记战场关键位置 |
| 经济/科技模拟 | 开启 diffusion；diffusion_fields = `["technology", "economy"]` |
| 种群生态模拟 | 关闭 explosion；diffusion_rate 调高模拟生物扩散 |
| 纯抽象策略 | 关闭 physics 全部子系统；仅保留 ODE |
| 大规模 N 实体 | collision 自动切换空间哈希（>150 实体），无需手动调整 |

---

### 模块上下文（ModuleContext）

文件：`algorithms/base.py`

模块之间通过 `ModuleContext` 交换数据，所有 field 为 numpy 数组：

```python
@dataclass
class ModuleContext:
    round_number: int          # 当前轮次
    dt: float = 1.0            # 时间步长
    arrays: dict = {}          # 指标数组 {metric_name: np.ndarray}
    spatial: SpatialState      # 3D 空间状态
    diffusion_fields: list = []# 需扩散的指标名
    metadata: dict = {}        # 任意元数据（如 trigger_explosion）
```

`SpatialState` 包含 N 个实体的 positions / velocities / masses / radii / forces，均为 `float64` 数组。

### 模块链工厂（module_utils）

文件：`algorithms/module_utils.py`

| 函数 | 功能 |
|------|------|
| `build_module_chain(rule_engine)` | 从规则包 `modules` 段创建 [ODEModule, PhysicsModule] 链 |
| `build_context(states, rule_engine, entity_ids, round_number)` | EntityState → ModuleContext 转换 |
| `apply_context_results(ctx, states, entity_ids, rule_engine)` | ModuleContext → EntityState 写回 |

工厂配置优先级：**规则包 `modules` 段 > 内置预设匹配**。

内置预设匹配规则（`ode_preset_map`）：
- `fatigue`/`supply`/`pollution` → 对应同名预设
- `population`/`economy`/`market_share`/`brand` → `logistic`
- `cash_flow` → `decay`

若规则包的 `modules` 段完全不存在或为空，工厂仍会创建模块链并应用预设匹配 —— 行为等价于所有量化推演默认启用 ODE + Physics。

---

## 打包与发布（Windows）

桌面安装包由 PyInstaller 后端 + Tauri NSIS 打包组成：

```bash
# 1. 打包前端
cd apps/strategy-forge
npm run build

# 2. 打包后端（项目根目录，onedir）
python -m PyInstaller strategy-forge-backend.spec --noconfirm

# 3. 拷入 data/rule/ 规则文件
#    data/rule/  →  dist/strategy-forge-backend/data/rule/

# 4. 拷入 Tauri 资源目录
#    dist/strategy-forge-backend/  →  apps/strategy-forge/src-tauri/resources/strategy-forge-backend/

# 5. 构建桌面应用与安装包
cd apps/strategy-forge
npx tauri build --bundles nsis
```

产物：
- 应用：`apps/strategy-forge/src-tauri/target/release/strategy-forge.exe`
- 安装包：`apps/strategy-forge/src-tauri/target/release/bundle/nsis/StrategyForge_0.1.0_x64-setup.exe`
- 建议收集：将以上两个产物 + 后端 onedir 目录统一复制到 `release/` 文件夹。

CI/CD 自动构建脚本（AVX2/BMI2 优化 + UPX 压缩）：`.github/workflows/release.yml`

运行时 Tauri 壳会自动以子进程拉起后端（`strategy-forge-backend.exe serve`），关闭窗口最小化到系统托盘，退出时清理后端进程。

> `strategy-forge-backend.spec` 排除了 scipy 的重子模块（sparse/linalg/special 等）及 UPX 压缩排除（`VCRUNTIME140.dll`, `python3.dll`, `*.pyd`）。对外分发前请清理 `data/forge_config.json`（可能含 API Key）与历史会话库。

---

## 主要 API（前缀 `/api/forge`）

### 会话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传种子材料文件 |
| POST | `/session` | 创建会话 |
| GET | `/sessions` | 会话列表 |
| GET | `/session/{id}` | 会话详情 |
| DELETE | `/session/{id}` | 删除会话（连带清理 Kuzu/ LanceDB） |
| DELETE | `/session/{id}/force` | 强制删除（含运行中会话） |

### 推演控制

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/session/{id}/start` | 启动推演（新建或从暂停恢复） |
| POST | `/session/{id}/pause` | 暂停推演（保存快照，状态变为 `paused`） |
| POST | `/session/{id}/resume` | 继续推演（从快照恢复） |
| POST | `/session/{id}/start/cancel` | 取消推演任务 |
| POST | `/session/{id}/settings` | 推演级设置（多动作、领域等） |
| POST | `/session/{id}/pre-goal` | 设定推演前不可变目标 |
| POST | `/session/{id}/intervene` | 实时干预注入 |

### 数据查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/session/{id}/graph` | 知识图谱（节点 + 边） |
| GET | `/session/{id}/timeline` | 智能体行动时间线 |
| GET | `/session/{id}/causal` | 因果子图（TARGETS / CAUSED） |
| GET | `/session/{id}/report` | 推演报告 |
| GET | `/session/{id}/logs` | 会话日志 |
| GET | `/session/{id}/tokens` | Token 统计 |
| GET | `/session/{id}/stream` | SSE 实时事件流 |

### 策略优化器

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/session/{id}/optimize` | 启动蒙特卡洛优化 |
| POST | `/session/{id}/optimize/cancel` | 取消优化 |
| GET | `/session/{id}/optimize/result` | 优化进度与结果轮询 |

### 配置与规则

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/domains` | 可用规则包领域列表 |
| POST | `/rules/upload` | 上传自定义规则包 |
| GET/POST | `/config/llm` | LLM 端点与模型配置 |
| GET/POST | `/config/embedding` | 嵌入端点与模型配置 |
| GET | `/config/providers` | 厂商目录 |
| POST | `/config/list-models` | 测试获取模型列表 |
| POST | `/config/test-connection` | 测试端点连通性 |
| POST | `/config/reload` | 重新加载配置 |

---

## 项目结构

```
StrategyForge/
├── src/strategy_forge/
│   ├── algorithms/       # 通用算法模块（ODE / Physics / 基类 / 模块链工厂）
│   ├── core/             # 配置 / Provider 注册表 / LLM 适配 / Token 统计 / 分块器
│   ├── engine/           # 五阶段流水线 + 规则引擎 + 优化器 + 预处理器(LanceDB) + 推理器
│   ├── storage/          # SQLite 会话库 + Kuzu 图库
│   └── api/              # FastAPI 路由 + SSE 事件流 + 配置路由
├── apps/strategy-forge/          # Tauri 2 桌面应用 (React 18 + 3D 图)
│   └── src-tauri/                # Rust 壳（子进程管理 + 系统托盘 + NSIS 打包）
├── scripts/                      # 端到端测试脚本（依赖 LM Studio）
├── tests/                        # 单元测试 + 功能测试
├── data/                         # 运行期数据（SQLite / Kuzu / LanceDB / forge_config.json）
│   └── rule/                     # 内置规则包 + 自定义规则包目录
├── strategy-forge-backend.spec   # PyInstaller 打包配置
├── release/                      # 打包产物（exe + 安装包 + 后端目录）
├── run.py                        # 开发启动器
└── pyproject.toml
```

---

## 测试

### 单元测试

```bash
python -m pytest tests/ -v
```

### 端到端测试（需本地 LM Studio 运行对话 + 嵌入模型）

`scripts/` 下提供连接本地 LM Studio 的验证脚本：

```bash
python scripts/test_algorithms_lmstudio.py         # ODE + Physics + Token + 量化全流程（5 项）
python scripts/test_pause_resume_lmstudio.py       # 暂停/恢复流程（取消信号响应 + 快照保存/载入）
python scripts/test_quantified_lmstudio.py         # 量化推演闭环
python scripts/test_multi_action_lmstudio.py       # 多动作资源分配
python scripts/test_optimizer_lmstudio.py          # 蒙特卡洛优化器
python scripts/test_optimizer_report_lmstudio.py   # 优化器 + 报告生成
python scripts/test_graph_fulllink_lmstudio.py     # Kuzu 关系反哺 + 时序 + 因果链
python scripts/test_lancedb_fulllink_lmstudio.py   # LanceDB 全文检索链路
python scripts/test_geo_strategy.py               # 地缘战略全流程
```

运行前请确保本地 LM Studio 已启动并加载对话与嵌入模型。
