# StrategyForge — 多智能体战略推演引擎

**把战略决策变成可计算、可复现、可优化、可解释的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具。以一段「种子材料」为输入，自动构建知识图谱与智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"的闭环并行推演多轮，并能对多套策略做蒙特卡洛对比择优。支持**量化模式**（规则包驱动、数值动力学、阈值淘汰）和**叙事模式**（自由推理、目标收敛、因果链保障）两种推演模式。以「Python 后端 + Tauri 桌面应用」形态交付，可打包为独立安装包离线运行。

---

## 目录

1. [架构概览](#架构概览)
2. [核心特性](#核心特性)
3. [实体治理架构](#实体治理架构)
4. [六阶段推演流水线](#六阶段推演流水线)
5. [智能体决策系统](#智能体决策系统)
6. [信息生态](#信息生态)
7. [算法模块](#算法模块)
8. [规则包体系](#规则包体系)
9. [策略优化器](#策略优化器)
10. [知识图谱与因果链](#知识图谱与因果链)
11. [安装与运行](#安装与运行)
12. [配置参考](#配置参考)
13. [API 参考](#api-参考)
14. [开发指南](#开发指南)

---

## 架构概览

```
                    ┌──────────────┐
用户输入种子材料 →   │ DeductionEngine │ ← 会话管理 / SSE 事件
                    └──────┬───────┘
                           │
             ┌─────────────┴─────────────┐
             │  DeductionOrchestrator     │
             │  (5-Phase Pipeline)        │
             │  Phase 1: 本体生成          │
             │  Phase 1.5: 量化(规则包)    │
             │  Phase 2: GraphRAG 图谱     │
             │  Phase 2.5: EntityRegistry  │ ← 确定性实体分类
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

                    ┌──────────────────────────────────┐
                    │     规则包体系 (8 域 + 继承)       │
                    │  7 垂直域 → geo_strategy 全域继承 │
                    └──────────────────────────────────┘
```

---

## 核心特性

### 实体治理：确定性分类 + LLM 审核

实体分类由纯 Python 代码规则驱动（零 LLM 参与决策），通过 **EntityRegistry** 模块实现：
- 同一份种子材料永远产出同一份 agent 列表（确定性、可复现）
- 代码规则矩阵：类型排除 → 二元词检测 → 领域词保留 → 自适应阈值
- LLM 审核机制（`FORGE_LLM_REVIEW=1`）：审核边缘实体，纠正结构化错误（军队编制/职务头衔/核心人物），只做增补不做终审
- 所有分类决策有完整日志可追溯

### 规则继承体系

geo_strategy 全域模块通过 `inherited_from` 从子域（military/business/politics/tech/info_war）自动继承 metrics/actions/effects，消除 ~600 行重复配置。修改子域自动同步全域，一处修改、全域生效。

### 模拟模式

| 模式 | 决策方式 | 胜负判定 | 适用场景 |
|------|---------|---------|---------|
| **量化模式** | 规则包驱动（8 域 + 继承 + 自定义），FSM + LLM 分层决策 | 阈值淘汰 / 加权综合判定 | 军事、商业、政治推演 |
| **叙事模式** | LLM 自由推理 + 三幕节拍 + 人格演化 | 收敛裁判 + 证据一致性收束 | 趋势推演、路径探索、创意写作 |
| **多动作分配** | 每轮预算按权重分配多个动作 | 同量化/叙事 | 混合策略、多线博弈 |

---

## 实体治理架构

### EntityRegistry — 唯一权威数据源

```
Kuzu 图谱  ──→  build_registry(graph, ontology, domain)
      │              │
      │         _classify_one() × N  ← 纯 Python 代码规则
      │              │
      │         LLM Review (可选)    ← 审核边缘实体
      │              │
      ▼              ▼
   注册表 {name → RegisteredEntity(decision=KEEP|DISCARD, reason=..., parent=...)}

     ├──  agent_factory  ← get_kept() → persona 生成
     ├──  reporter       ← 实体统计
     └──  optimizer      ← 量化实体池
```

### 分类规则矩阵（按优先级）

1. 类型排除：Location/Document/Event/Concept 等 → DISCARD
2. 二元关系词：俄乌/中美/A与B/中美关系 → DISCARD
3. 领域保留词（`extra_keep_words`）：政党/议会/集团等 → KEEP（覆盖后续规则）
4. 职务头衔：总统/总理/司令等 → DISCARD
5. 军队编制：X舰队/X战区/X军 → DISCARD
6. 政府部门：国防部/财政部/央行等 → DISCARD
7. 集合概念：阵营/群体/板块等 → DISCARD
8. 领域排除词（`extra_discard_words`）：公报/阵地等 → DISCARD
9. 自适应阈值：Person ≥ total/domain_factor, Org ≥ total/domain_factor×2

### 领域配置（domain_prompts.json）

每个领域可配置 `registry_tweak`：
- `threshold_factor`：阈值因子（military=25, politics=30, ecology=60）
- `extra_keep_words`：强制保留词
- `extra_discard_words`：强制排除词
- `agent_domain_role`：注入 persona 生成
- `strategic_context`：注入量化推演决策

---

## 六阶段推演流水线

```
Phase 1: 本体生成    → LLM 提取实体/关系类型定义（SHA256 缓存）
Phase 1.5: 量化      → 领域检测 / 规则包加载 / 种子指标 LLM 提取
Phase 2: GraphRAG    → jieba 分词 + 语义分块 + LanceDB 索引 + sorter(别名/层级)
Phase 2.5: EntityRegistry → 代码规则分类 + LLM 审核（可选）
Phase 3: 智能体工厂   → 从注册表读取 KEPT 实体 → LLM 生成 persona
Phase 4: 并行模拟    → 每轮：FSM 分流 → 并发 LLM → resolve_round → ODE+Physics
                      每 N 轮：目标收敛裁判 + 人格反思（事件驱动）
Phase 5: 报告生成    → 量化和叙事双模式，弧线事件采样 + 因果锚点
```

### Phase 2 细节

1. 语义分块：TextChunker 递归分层分割源文本
2. 实体发现：纯 jieba 分词（无 LLM），确定性输出
3. LanceDB 索引：批量向量嵌入 + FTS 全文索引
4. 图谱提取：全量 chunk-pass（无 entity-driven），温度 0
5. Sorter：只输出别名和层级关系（不再参与分类决策）

### Phase 4 细节

```
每轮：
  1. FSM 分流：非命令态 agent 走确定性动作（0 LLM 调用）
  2. 并发 LLM：命令态 agent asyncio.gather 并行决策
  3. resolve_round：批量计算所有 self/target/conditional effects
  4. 批量应用 deltas + auto effects + delay effects
  5. 声誉更新 + 谍报处理 + 事件分发（信任度驱动延迟/失真）
  6. 算法模块链：opinion_dynamics → ODE → physics
  7. 人格反思：环境漂移>5 / 关系变化 / 指标告急 / 逾6轮保护
  8. 可选叙事 + 态势快照生成
```

---

## 智能体决策系统

### FSM + LLM 分层决策

```
Agent 状态         决策路径            LLM 调用
──────────────────────────────────────────────
patrol/retreat    FSM 确定性动作        零
engage/combat     StrategicReasoner     每次
（命令态）        LLM prompt 构建
```

FSM 优势：非命令态 agent 完全由确定性规则驱动，实测可节省 60-80% 的 LLM 请求。

FSM 支持对手感知（v2.1+）：通过 `opponent.strength` / `opponent.morale` 等条件读取敌方实际指标值，实现动态博弈反制。

### 规则继承体系

geo_strategy 全域模块通过 `inherited_from` 自动继承：
```
military ─┐
business ─┤
politics ─┼──→ geo_strategy（28 metrics, 40 actions, 跨域复合动作）
tech ─────┤         仅定义：跨域联动 + 全局条件效应 + 扩散 + 淘汰加权
info_war ─┘
```

### 六大核心维度

| 域 | 指标数 | 新增指标 | 关键特性 |
|----|--------|---------|---------|
| ⚔️ military | 5 | — | spatial combat、distance_to_enemy FSM |
| 📊 business | 6 | — | brand competitive_logistic、供应链保护 |
| 🏛️ politics | 6 | — | support_rate competitive、立法博弈 |
| 🔬 tech | 5 | — | tech_lead+talent_pool 双 competitive |
| 📰 info_war | 4 | — | public_trust+polarization 双 competitive |
| 🌐 geo_strategy | **28** | energy_resource, grain_resource, global_reputation, alliance_trust | **全域继承 + 跨域复合行动 + delay_effects + opponent FSM** |

---

## 信息生态

### 第一层：信息不对称

量化模式——信任度驱动的情报质量：
```
信任度 [-5, +5] → 延迟 [4, 0] 轮 + 失真 [0%, 30%]
```

叙事模式——私密行动可见性隔离：秘密录音、私下会面等自动标记 visibility=private，非参与者无法感知。

### 第二层：声誉积累

攻击/包围/制裁 → trust -2.5×intensity；外交/合作/投资 → trust +1.5×intensity。每一轮交互自动更新 trust matrix。

### 第三层：人格动态化

**触发条件**（共享闸门）：
- 环境累积剧变：任一维度漂移 >5 或累计漂移 >12
- 关系网络变化：盟友/对手集合发生变化
- 量化模式补充：任意指标逼近淘汰线
- 长期无反思保护：超过 6 轮

**执行逻辑**：调用 LLM（temperature=0.3）生成行为准则（≤20 字），写入 system_prompt_extra，FIFO 保留 3 条，注入后续每轮决策 prompt。被背叛的角色真的会变多疑并拒绝后续合作。

---

## 算法模块

### ODE 引擎

8 类微分方程：decay / logistic / fatigue_recovery / supply_consumption / pollution_spread / resource_depletion / competitive_logistic / cash_flow_dynamics。scipy RK45 自适应步长，降级为 Euler 冻结态两步法。

### 3D 物理引擎

刚体力学（重力、阻尼）、自适应碰撞检测（N>150 时空间哈希 O(N)）、各向同性高斯扩散、径向冲击波。

### 观点动力学（HK 模型）

Hegselmann-Krause 有界置信模型。geo_strategy epsilon=0.07（最难动摇），urban epsilon=0.25（最易相互影响）。

### 有限状态机（FSM）

每域定义专用状态循环。支持：streak 历史条件、虚拟空间度量（distance_to_enemy/ally）、**对手指标感知**（opponent.strength）、自动敌友划分、FSM override。

---

## 规则包体系

### 结构

```json
{
  "domain_key": {
    "name": "显示名称",
    "inherited_from": ["ancestor_domain", ...],  // 继承父域规则
    "metrics": ["metric1", ...],
    "initial_metrics": {},
    "thresholds": {},
    "actions": [],
    "self_effects": {},
    "target_effects": {},
    "conditional_effects": {},
    "delay_effects": {},
    "auto_effects": {},
    "modules": {
      "ode_engine": {},
      "physics_engine": {},
      "opinion_dynamics": {},
      "finite_state_machine": {},
      "pipeline": { "order": [] }
    }
  }
}
```

### 递延反噬（delay_effects）

激进单边动作配置滞后 1-3 轮隐形负收益：
```
military_offensive → supply_chain: -5, global_reputation: -5 (delay 2 rounds)
trade_warfare → cash_flow: -8, intl_relations: -5 (delay 3 rounds)
```

实现"短期获利、长期有代价"的真实博弈规律。

---

## 策略优化器

蒙特卡洛多方案并行对比：
1. 用户定义 M 个候选策略方案
2. 每方案运行 N 次独立模拟（随机种子 + 温度抖动）
3. 量化模式使用 RuleEngine.judge() 客观判胜负
4. 统计分析：成功率 / 胜率 / CI95 / 成本
5. 帕累托前沿推荐最优方案

---

## 知识图谱与因果链

### Kuzu 图数据库

```
Node Tables:
  Entity(id, name, type, description)
  Agent(id, name, persona, background, goals)
  Event(id, description, event_type, timestamp, ...)

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
| `deduction_events_{id}` | 模拟事件记忆（动态，含 visibility） | 向量 + FTS + 观察者可见性过滤 |

---

## 安装与运行

### 前置依赖

- Python ≥ 3.11
- Node.js ≥ 20
- Rust toolchain（仅打包时需要）
- LM Studio / Ollama / OpenAI API（LLM 服务端）

### 快速开始

```bash
pip install -e .
python run.py
# 访问 http://localhost:5173（开发模式）或 http://localhost:8000（API）
```

### 打包安装包（Windows）

```bash
# 1. PyInstaller 打包后端
python -m PyInstaller strategy-forge-backend.spec --noconfirm

# 2. 同步后端到 Tauri resources
$res = "apps\strategy-forge\src-tauri\resources\strategy-forge-backend"
Remove-Item "$res\_internal","$res\strategy-forge-backend.exe" -Recurse -Force
Copy-Item "dist\strategy-forge-backend\strategy-forge-backend.exe","dist\strategy-forge-backend\_internal" -Destination $res -Recurse -Force
Copy-Item "data\rule\rules.json" -Destination "$res\data\rule\rules.json" -Force
Copy-Item "data\rule\domain_prompts.json" -Destination "$res\data\rule\domain_prompts.json" -Force
Copy-Item "data\custom_dict\classic_names.txt" -Destination "$res\data\custom_dict\classic_names.txt" -Force

# 3. 构建前端 + Tauri + NSIS 安装包
cd apps\strategy-forge
npx tauri build --bundles nsis

# 4. 复制安装包到 release 目录
Copy-Item "apps\strategy-forge\src-tauri\target\release\bundle\nsis\StrategyForge_*_x64-setup.exe" -Destination "release\StrategyForge_Setup.exe" -Force
```

---

## 配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_PROVIDER` | `lmstudio` | LLM 提供商标识 |
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | LLM API 地址 |
| `FORGE_LLM_MODEL` | — | LLM 模型名称 |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | Embedding 模型 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认推演轮数 |
| `FORGE_MAX_AGENTS` | `10000` | 最大智能体数 |
| `FORGE_MAX_CONCURRENT` | `2` | 并发 LLM 调用上限 |
| `FORGE_RETRIEVE_TOP_K` | `5` | 语义检索返回数 |
| `FORGE_LLM_REVIEW` | `1` | LLM 审核开关（EntityRegistry 边缘实体审核） |
| `FORGE_CTX_LIMIT` | `262144` | LLM 上下文窗口上限 |
| `FORGE_REPORT_MAX_TOKENS` | `30000` | 报告输出上限 |

---

## API 参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/forge/session` | POST | 创建推演会话 |
| `/api/forge/sessions` | GET | 会话列表 |
| `/api/forge/session/{id}` | GET/DELETE | 会话详情 / 删除 |
| `/api/forge/session/{id}/start` | POST | 启动推演 |
| `/api/forge/session/{id}/pause` | POST | 暂停推演 |
| `/api/forge/session/{id}/resume` | POST | 恢复推演 |
| `/api/forge/session/{id}/intervene` | POST | 用户干预注入 |
| `/api/forge/session/{id}/pre-goal` | POST | 推演目标注入 |
| `/api/forge/session/{id}/optimize` | POST | 策略优化（蒙特卡洛） |
| `/api/forge/session/{id}/graph` | GET | 知识图谱数据 |
| `/api/forge/session/{id}/timeline` | GET | 行动时序 |
| `/api/forge/session/{id}/report` | GET | 推演报告 |
| `/api/forge/session/{id}/logs` | GET | 推演日志 |
| `/api/forge/session/{id}/tokens` | GET | Token 统计 |
| `/api/forge/session/{id}/stream` | GET | SSE 事件流 |
| `/api/forge/domains` | GET | 可用领域列表 |
| `/api/forge/config/llm` | GET/POST | LLM 配置 |
| `/api/forge/config/embedding` | GET/POST | Embedding 配置 |
| `/api/forge/config/engine` | GET/POST | 引擎参数配置 |

---

## 开发指南

### 项目结构

```
src/strategy_forge/
├── core/          配置、LLM 客户端、Token 计数器、分块器、规则模板
├── storage/       会话存储(SQLite)、图谱存储(Kuzu)
├── api/           路由、配置路由
├── engine/        引擎核心（EntityRegistry、5 阶段流水线 + 推理 + 模拟 + 报告 + 优化器）
└── algorithms/    算法模块（ODE、Physics、OpinionDynamics、FSM、Pipeline）

data/rule/         规则包 rules.json（8 域 + 继承）+ domain_prompts.json（领域提示词配置）
data/custom_dict/  jieba 自定义词典
apps/strategy-forge/   Tauri 前端（React/Vite/TypeScript + Tailwind CSS）
tests/             单元测试
```

---

## 许可证

StrategyForge 采用 [Apache License 2.0](LICENSE) 开源。
