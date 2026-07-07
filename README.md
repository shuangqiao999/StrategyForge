# StrategyForge — 多智能体战略推演引擎

**把战略决策变成可计算、可复现、可优化、可解释的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具。以一段「种子材料」为输入，自动构建知识图谱与智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"的闭环并行推演多轮，并能对多套策略做蒙特卡洛对比择优。以「Python 后端 + Tauri 桌面应用」形态交付，可打包为独立安装包离线运行。

---

## 目录

1. [架构概览](#架构概览)
2. [核心特性](#核心特性)
3. [六阶段推演流水线](#六阶段推演流水线)
4. [智能体决策系统](#智能体决策系统)
5. [信息生态（信息不对称 + 声誉 + 人格演化）](#信息生态)
6. [算法模块](#算法模块)
7. [规则包体系](#规则包体系)
8. [策略优化器](#策略优化器)
9. [知识图谱与因果链](#知识图谱与因果链)
10. [安装与运行](#安装与运行)
11. [配置参考](#配置参考)
12. [API 参考](#api-参考)
13. [开发指南](#开发指南)

---

## 架构概览

```
                     ┌──────────────┐
用户输入种子材料 →    │  DeductionEngine  │ ← 会话管理 / SSE 事件
                     └──────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │  DeductionOrchestrator     │
              │  (6-Phase Pipeline)        │
              │  Phase 1: 本体生成          │
              │  Phase 1.5: 量化(规则包)    │
              │  Phase 2: GraphRAG 图谱     │
              │  Phase 3: 智能体工厂        │
              │  Phase 4: 并行模拟           │
              │  Phase 5: 报告生成           │
              └─────────────┬─────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼────┐      ┌──────▼──────┐    ┌─────▼─────┐
    │  Kuzu   │      │  LanceDB    │    │  SQLite   │
    │ 图谱存储 │      │ 语义记忆     │    │ 会话存储   │
    └─────────┘      └─────────────┘    └───────────┘
         │                  │
    ┌────▼──────────────────▼────┐
    │     规则包体系 (8 域)       │
    │  military / business /      │
    │  politics / ecology /       │
    │  urban / tech / info_war / │
    │  geo_strategy              │
    └────────────────────────────┘
```

---

## 核心特性

### 五阶段推演流水线

本体生成 → GraphRAG 知识图谱 → 智能体工厂 → 并行模拟 → 报告生成，全程自动化。

### 推演控制

- **启动 / 暂停 / 继续**：任意轮次可暂停，进度持久化到 SQLite，重启后从断点继续。
- **状态实时同步**：SSE 事件驱动，界面实时显示当前阶段。
- **实时干预**：推演中注入用户指令，影响智能体决策方向。
- **不可变目标（pre-goal）**：为会话设定最终战略目标，贯穿全轮次。
- **轮数可配**：界面下拉框预设 5/10/15/20/30/50/100 轮，支持手动输入 1-100 轮。

### 模拟模式

| 模式 | 决策方式 | 胜负判定 | 适用场景 |
|------|---------|---------|---------|
| **量化模式** | 规则包驱动（8 域 + 自定义），FSM + LLM 分层决策 | 阈值淘汰 / 加权综合判定 | 军事、商业、政治推演 |
| **叙事模式** | LLM 自由推理 | LLM 主观判断 | 创意写作、文本推演 |
| **多动作分配** | 每轮预算按权重分配多个动作 | 同量化/叙事 | 混合策略、多线博弈 |

---

## 六阶段推演流水线

```
Phase 1: 本体生成    → LLM 提取实体/关系类型定义
Phase 1.5: 量化      → 领域检测 / 规则包加载 / 种子指标 LLM 提取
Phase 2: GraphRAG    → 语义分块 + LanceDB 索引 + 实体/关系抽取 + 情报排序
Phase 3: 智能体工厂   → LLM 生成 persona/background/goals + Kuzu 持久化
Phase 4: 并行模拟    → 每轮：FSM 分流 → 并发 LLM → resolve_round → ODE+Physics
Phase 5: 报告生成    → 结构化报告（叙事/风险/建议/结论）
```

### Phase 2 细节：GraphRAG 知识图谱

1. 语义分块：`TextChunker` 递归分层分割源文本
2. 实体提取：jieba POS + LLM 补充，按频率分高低频两档
3. LanceDB 索引：批量向量嵌入 + FTS 全文索引
4. 情报排序：`IntelSorter` LLM 分类实体 + `_apply_safety_net()` 确定性安全网（中英文关键词）
5. 别名合并：Kuzu `merge_alias_nodes()` 重连关系边

### Phase 4 细节：并行模拟循环

```
每轮：
  1. FSM 分流：非命令态 agent 走确定性动作（0 LLM 调用）
  2. 并发 LLM：命令态 agent asyncio.gather 并行决策
  3. resolve_round：批量计算所有 self/target/conditional effects
  4. 批量应用 deltas + auto effects + delay effects
  5. 声誉更新 + 谍报处理 + 事件分发（信任度驱动延迟/失真）
  6. 算法模块链：opinion_dynamics → ODE → physics
  7. 人格反思：每 5 轮或重大指标变化时触发 LLM 自我审视
  8. 可选叙事 + 态势快照生成
```

---

## 智能体决策系统

### FSM + LLM 分层决策

```
Agent 状态         决策路径            LLM 调用
─────────────────────────────────────────────
patrol/retreat    FSM 确定性动作        零
engage/combat     StrategicReasoner     每次
（命令态）        LLM prompt 构建
```

FSM 的优势：非命令态的 agent 完全由确定性规则驱动（如 patrol→combat 阈值、retreat→observe 恢复路径），**实测可节省 60-80% 的 LLM 请求**。

### 决策上下文（六维信息注入）

```
共享前缀（云端 token 缓存友好）:
  ├── 不可变目标
  ├── 可选行动目录
  ├── 地形与天气
  ├── 用户干预指令
  └── 近期局势 (per-agent, 信任度驱动延迟/失真)

Agent 私有段:
  ├── 人格 / 目标 / 行为准则(system_prompt_extra)
  ├── 当前量化状态 (EntityState.to_prompt_context)
  ├── 他方状态 (Top-K 排序)
  ├── 关系网络 (Kuzu 盟友/对手)
  ├── 静态背景召回 (LanceDB Path A)
  ├── 动态事件记忆 (LanceDB Path B)
  ├── 增强因果反馈 (多段落叙事复盘)
  └── 空间环境 (3D 位置/距离)
```

### 趋势感知

`EntityState.history` 重建多轮指标变化轨迹，注入 prompt 为 `多轮趋势: strength↓30 supply↓12`，让 LLM 感知 direction，而非仅看当前快照。

---

## 信息生态

StrategyForge 构建了一个完整的**信息不对称 + 声誉演化 + 人格动态化**三层生态模型。

### 第一层：信息不对称

```
信任度 [-5, +5] → 延迟 [4, 0] 轮 + 失真 [0%, 30%]
```

每个 agent 接收到的近期事件是**按信任度个性化过滤**的：
- 盟友（trust +4）：0 轮延迟、0% 失真——"军力-20"精确情报
- 中立（trust 0）：2 轮延迟、13% 失真——"军力约-17~-23"区间估计
- 敌对（trust -4）：4 轮延迟、30% 失真——"军力遭受重创"纯定性

事件内容按轮次累积衰减信息劣化（每轮 +5% 失真），长时间未交付的情报逐渐褪色。

### 第二层：声誉积累

```
攻击/包围/制裁 → trust -2.5×intensity
外交/合作/投资 → trust +1.5×intensity
```

每一轮的交互结果自动更新 trust matrix。agent 不仅接收信息不对称，而且**通过自身行为影响未来的信息获取质量**。背叛盟友会让情报管道永久受损。

### 第三层：人格动态化（记忆内省）

**触发条件**：每 5 轮或单一指标累计变化超过 25 点时触发。

**执行逻辑**：
1. 收集 agent 的近期经历（指标变化、因果反馈、风险信号）
2. 调用轻量 LLM（temperature=0.3）生成一条行为准则（≤20 字）
3. 写入 `system_prompt_extra`（不修改原始 persona）
4. 准则可累积叠加——多轮演化后形成个性化行为链

**实测效果**（geo_strategy 8 轮）：
- Alpha 从"军事扩张主义者" 演化出："疲劳度告急时立即停止扩张并优先休整；国际关系告急时立即暂停扩张并外交缓和"
- Bravo 演化出："疲劳度告急时优先休整以保长期稳定；军力补给双降时优先休整以保稳定"
- Charlie 演化出："疲劳告急时立即强制休息恢复精力"

### 第四层：谍报行动

6 个领域已配置 `intel_gather` 动作。agent 投入资源进行情报搜集后，对特定目标获得信息优势——延迟降低、失真归零。

---

## 算法模块

规则包的 `modules.pipeline.order` 控制算法模块的执行序列，`IS_FINALIZER` 标记确保分析模块先于写入模块运行。

### ODE 引擎（`ODE_PRESETS`）

| 方程 | 数学形式 | 适用指标 |
|------|---------|---------|
| `decay` | dy/dt = -rate × y | 自然衰减（信任度、团结度） |
| `logistic` | dy/dt = rate × y × (1 - y/K) | 受限增长（人口、经济） |
| `fatigue_recovery` | dy/dt = -rate × √y | 疲劳恢复（越累恢复越快） |
| `supply_consumption` | dy/dt = -base - strength×factor | 供给消耗（含钳制防负） |
| `pollution_spread` | dy/dt = factories - greens - decay | 污染扩散（多源） |
| `resource_depletion` | dy/dt = -rate × population | 资源消耗（人口驱动） |
| `competitive_logistic` | growth + diffusion + crowding | 零和技术/市场博弈 |
| `cash_flow_dynamics` | decay + supply_chain/tech protection | 现金流（含保护机制） |

**积分方法**：scipy RK45 自适应步长（可用时），降级为 Euler 冻结态两步法（sub_steps=8，含快照恢复）。

### 3D 物理引擎

| 子系统 | 描述 |
|--------|------|
| `dynamics` | 刚体力学（重力、阻尼、速度上限钳制） |
| `collision` | 自适应碰撞检测（N>150 时空间哈希 O(N) 降维） |
| `diffusion` | 各向同性高斯扩散（支持边界条件 absorb/reflect） |
| `explosion` | 径向冲击波（静态配置 + 运行时注入） |

### 观点动力学（HK 模型）

Hegselmann-Krause 有界置信模型：每个实体向"观点差异 < epsilon" 的邻居靠拢。

| 域 | epsilon | 设计意图 |
|----|---------|---------|
| geo_strategy | 0.07 | 大国博弈立场固化，最难动摇 |
| military | 0.12 | 士气由战果驱动，同质性低 |
| politics | 0.12 | 支持率由政绩决定 |
| info_war | 0.15 | 舆论战手段多样，保持分化 |
| tech / business | 0.20 | 品牌/技术趋势半公开传播 |
| urban | 0.25 | 市民满意度最易相互影响 |

### 有限状态机（FSM）

每个域定义专用的状态循环，非命令态自动产出确定性动作，命令态交给 LLM。支持：
- **streak 历史条件**（"连续 N 轮满足"防抖动）
- **虚拟空间度量**（distance_to_enemy / distance_to_ally）
- **自动敌友划分**（按 polarization 极化度或"全部其他方为敌"）
- **FSM override**（用户强制指定 agent 动作）

---

## 规则包体系

### 内置 8 域

| 域 | 指标数 | ODE 方程数 | FSM 状态 | 关键特性 |
|----|--------|-----------|---------|---------|
| ⚔️ military | 5 | 2 | patrol/combat/retreat/defend | 空间图 opinion_dynamics、物理碰撞、distance_to_enemy FSM |
| 📊 business | 6 | 4 | active/retrench/defensive | brand competitive_logistic、cash_flow_dynamics 含供应链保护 |
| 🏛️ politics | 5 | 3 | stable/campaign/defensive | support_rate competitive_logistic、反宣传效果 |
| 🌿 ecology | 5 | 4 | stable/intervention/expansion | population competitive_logistic、污染物理扩散、diffusion_boundary |
| 🏙️ urban | 5 | 2 | maintain/develop/austerity | 城市满意度 opinion_dynamics |
| 🔬 tech | 5 | 2 | research/launch | tech_lead+talent_pool 双 competitive |
| 📰 info_war | 4 | 2 | monitor/offensive/defensive | public_trust+polarization 双 competitive |
| 🌐 geo_strategy | **12** | 5 | observe/engage/retreat | **最复杂域**：跨域联动 + weighted HK + elimination 加权淘汰 + 所有特征全开 |

### 规则包结构

```json
{
  "domain_key": {
    "metrics": ["metric1", "metric2", ...],
    "initial_metrics": {},      // LLM 种子提取可覆盖
    "thresholds": {},           // 淘汰判定线
    "actions": [],              // 可用行动列表
    "self_effects": {},         // {action: {metric: delta}}
    "target_effects": {},       // {action: {target_metric: delta}}
    "conditional_effects": {},  // 状态依赖条件效果
    "delay_effects": {},        // N 轮延迟结算效果
    "auto_effects": {},         // 每轮被动条件效果
    "modules": {
      "ode_engine": { "equations": {}, "params": {} },
      "physics_engine": {},
      "opinion_dynamics": {},
      "finite_state_machine": {},
      "pipeline": { "order": [] }
    }
  }
}
```

---

## 策略优化器

蒙特卡洛多方案并行对比：

1. 用户定义 M 个候选策略方案（不同战略指令）
2. 每方案运行 N 次独立模拟（随机种子 + 温度抖动）
3. 量化模式使用 `RuleEngine.judge()` 客观判胜负
4. 统计分析：成功率 / 胜率 / CI95 / 成本
5. 帕累托前沿：按优化目标（max_win_rate / min_cost / balanced）推荐最优方案

---

## 知识图谱与因果链

### Kuzu 图数据库

```
Node Tables:
  Entity(id, name, type, description)
  Agent(id, name, persona, background, goals)
  Event(id, description, event_type, timestamp, agent_id, round, target_id, effect, driver)

Relationship Tables:
  RELATES (Entity → Entity)   relation, weight, evidence
  ACTED   (Agent → Event)     action, timestamp
  TARGETS (Event → Entity)
  CAUSED  (Event → Entity)    metric, amount       ← 确定性因果归因
```

### LanceDB 语义记忆

| 表 | 用途 | 检索方式 |
|----|------|---------|
| `deduction_chunks_{id}` | 原著语义分块（静态） | 向量 + FTS 混合检索 |
| `deduction_events_{id}` | 模拟事件记忆（动态） | 向量 + FTS 混合（RRF 融合排序）+ per-round 缓存 |

---

## 安装与运行

### 前置依赖

- Python ≥ 3.11
- Node.js ≥ 20
- Rust toolchain（仅打包时需要）
- LM Studio / Ollama / OpenAI API（LLM 服务端）

### 快速开始

```bash
# 安装依赖
pip install -e .

# 启动开发服务器
python run.py
# 访问 http://localhost:5173（开发模式）或 http://localhost:8000（API）
```

### 打包安装包（Windows）

```bash
# 1. PyInstaller 打包后端
python -m PyInstaller strategy-forge-backend.spec --noconfirm

# 2. 拷贝规则包
mkdir -p dist/strategy-forge-backend/data/rule/custom
cp data/rule/rules.json dist/strategy-forge-backend/data/rule/

# 3. 拷贝到 Tauri 资源目录
cp -r dist/strategy-forge-backend apps/strategy-forge/src-tauri/resources/

# 4. Tauri NSIS 打包
cd apps/strategy-forge && npx tauri build --bundles nsis

# 输出: apps/strategy-forge/src-tauri/target/release/bundle/nsis/StrategyForge_*.exe
```

---

## 配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_PROVIDER` | `lmstudio` | LLM 提供商标识 |
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | LLM API 地址 |
| `FORGE_LLM_MODEL` | — | LLM 模型名称 |
| `FORGE_LLM_KEY` | — | API Key（本地可为空） |
| `FORGE_EMBED_BASE` | — | Embedding API 地址 |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | Embedding 模型 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认推演轮数 |
| `FORGE_MAX_AGENTS` | `10000` | 最大智能体数 |
| `FORGE_CANDIDATE_COUNT` | `3` | 候选动作生成数 |
| `FORGE_LLM_TEMPERATURE` | `0.6` | LLM 默认温度 |
| `FORGE_MAX_CONCURRENT` | `2` | 并发 LLM 调用上限 |
| `FORGE_RETRIEVE_TOP_K` | `5` | 语义检索返回数 |
| `FORGE_SIMILARITY_THRESHOLD` | `0.4` | 检索相似度阈值 |
| `FORGE_INTEL_SAFETY_NET` | `1` | 实体安全网开关 |
| `FORGE_RECALL_REL_BOOST` | `0` | 关系邻居召回增强 |
| `FORGE_EVENT_HYBRID` | `false` | LanceDB 混合检索 |
| `FORGE_LLM_TIMEOUT` | `240` | 总超时（秒） |
| `FORGE_LLM_CONNECT_TIMEOUT` | `15` | 连接超时（秒） |
| `FORGE_LLM_GENERATION_TIMEOUT` | `180` | 生成超时（秒） |

---

## API 参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/forge/session` | POST | 创建推演会话 |
| `/api/forge/session/{id}/start` | POST | 启动推演 |
| `/api/forge/session/{id}/start/cancel` | POST | 取消推演 |
| `/api/forge/session/{id}/pause` | POST | 暂停推演 |
| `/api/forge/session/{id}/resume` | POST | 恢复推演 |
| `/api/forge/session/{id}/intervene` | POST | 用户干预注入 |
| `/api/forge/session/{id}/optimize` | POST | 策略优化 |
| `/api/forge/session/{id}/graph` | GET | 知识图谱数据 |
| `/api/forge/session/{id}/report` | GET | 推演报告 |
| `/api/forge/session/{id}/tokens` | GET | Token 统计 |
| `/api/forge/session/{id}/stream` | GET | SSE 事件流 |
| `/api/forge/domains` | GET | 可用领域列表 |
| `/api/forge/config/llm` | GET/POST | LLM 配置 |
| `/api/forge/config/embedding` | GET/POST | Embedding 配置 |
| `/api/forge/config/test-connection` | POST | 连接测试 |

---

## 开发指南

### 项目结构

```
src/strategy_forge/
├── core/          配置、LLM 客户端、Token 计数器、分块器、规则模板
├── storage/       会话存储(SQLite)、图谱存储(Kuzu)
├── api/           路由、配置路由
├── engine/        引擎核心（6 阶段流水线 + 推理 + 模拟 + 报告 + 优化器）
└── algorithms/    算法模块（ODE、Physics、OpinionDynamics、FSM、Pipeline）

data/rule/        内置规则包 rules.json（8 域）
apps/strategy-forge/   Tauri 前端（React/Vite/TypeScript）
tests/             单元测试
scripts/           LM Studio 集成测试脚本
```

### 添加新域

1. 在 `data/rule/rules.json` 中添加新域条目（参考 `geo_strategy` 作为最完整模板）
2. 定义 `metrics`、`actions`、`self_effects`、`target_effects`
3. 配置 `modules.ode_engine` 指标→方程映射
4. 配置 `modules.finite_state_machine` 状态循环
5. 配置 `modules.pipeline.order` 模块执行序列
6. 无需修改任何 Python 代码——规则包完全数据驱动

### 运行测试

```bash
# 单元测试（无 LLM）
python -m pytest tests/ --ignore=tests/functional

# LM Studio 集成测试（需本地 LLM）
python scripts/test_geo_tuning_lmstudio.py
python scripts/test_reflection_rollout_lmstudio.py
python scripts/test_causal_propagation_lmstudio.py
python scripts/test_report_quality_lmstudio.py
```
