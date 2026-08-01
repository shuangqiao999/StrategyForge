# StrategyForge — 多智能体战略推演引擎

**把战略决策变成可计算、可复现、可优化的实验。**

StrategyForge 是一款本地优先的多智能体战略推演工具。以一段种子材料（文本/文档）为输入，自动构建知识图谱、生成智能体人格，按"决策 → 数值计算 → 反馈 → 客观判胜负"闭环并行推演多轮，支持蒙特卡洛策略择优。以 **Python 后端 + Tauri 桌面应用** 形态交付，可打包为独立安装包离线运行。

---

## 目录

1. [核心特性](#核心特性)
2. [推演流水线](#推演流水线)
3. [实体注册中心（EntityRegistry）](#实体注册中心entityregistry)
4. [算法引擎](#算法引擎)
5. [规则包体系](#规则包体系)
6. [策略优化器](#策略优化器)
7. [信息生态与人格演化](#信息生态与人格演化)
8. [存储架构](#存储架构)
9. [安装与运行](#安装与运行)
10. [配置参考](#配置参考)
11. [API 参考](#api-参考)
12. [项目结构](#项目结构)

---

## 核心特性

### 三大推演模式

| 模式 | 决策引擎 | 胜负判定 | 适用场景 |
|------|---------|---------|---------|
| **量化模式** | 规则包驱动 + FSM + LLM 分层决策 | 阈值淘汰 / 加权综合判负 | 军事、商业、政治对抗 |
| **叙事模式** | LLM 自由推理 + 人格演化 | 收敛裁判 + 因果链收束 | 趋势探索、创意推演 |
| **多动作分配** | 每轮预算按权重分配多个动作 | 同量化/叙事 | 混合策略、多线博弈 |

### 规则包继承体系

`geo_strategy` 通过 `inherited_from` 自动继承 7 个子域（military / business / politics / tech / info_war / ecology / urban）的全部 metrics / actions / effects，消除重复配置。修改子域 → 全域自动同步。

### 通域实体识别

EntityRegistry 通过 **DomainAdapter + SemanticMediator** 架构实现领域无关的实体分类。所有领域判定规则外置为 `data/domain_adapters/*.yaml` 配置文件，新增领域只需添加 YAML、无需改代码。支持自动域探测，未知领域走通用中立适配器兜底。

### 信息不对称 + 人格动态化

信任度驱动情报延迟与失真，声誉基于行动自动更新。角色人格在满足触发条件（环境剧变 / 关系变化 / 指标告急）时通过 LLM 反思动态演化。

---

## 推演流水线

```
Phase 1: 本体生成     — LLM 提取实体/关系类型定义
Phase 1.5: 量化加载   — 领域检测 → 规则包加载 → 种子指标 LLM 提取
Phase 2: GraphRAG     — jieba 分词 → 语义分块 → LanceDB 索引 → 图谱构建 → 情报整理
Phase 3: 智能体工厂   — EntityRegistry 实体注册 → Agent Factory 人设生成
Phase 4: 并行模拟     — 每轮：FSM 分流 → 并发 LLM → 批量结算 → ODE/Physics → 人格反思
Phase 5: 报告生成     — 量化/叙事双模式，弧线事件采样 + 因果归因
```

### Phase 4 每轮流程

```
1. FSM 分流：非命令态 agent 走确定性动作（0 LLM）
2. 并发 LLM：命令态 agent asyncio.gather 并行决策
3. resolve_round：批量计算 self / target / conditional / delay effects
4. 算法模块链：opinion_dynamics → ODE → physics → auto_effects
5. 声誉更新 + 情报传播（信任度驱动延迟/失真）
6. 人格反思：环境漂移 > 5 / 关系变化 / 指标告急 / 逾 6 轮
7. 态势快照 + 可选叙事生成
```

---

## 实体注册中心（EntityRegistry）

### 架构

```
                     build_registry()
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       Layer 1         Layer 2        Layer 3
      (归一化)        (分级判定)      (交叉裁决)
            │             │             │
            ▼             ▼             ▼
     快速/分片路径    10实体批量      哈希缓存
     LLM 归一化      Parallel LLM    别名预合并
            │             │          LLM 冗余检测
            │             │             │
            └──────┬──────┴──────┬──────┘
                   ▼             ▼
           SemanticMediator   DomainAdapter
          (7大类基础类型)    (YAML 配置注入)
```

### Layer 1 — 实体归一化

- **快速路径**（短文本）：单次 LLM 全量归一化，>15 实体自动拆批并行
- **分片路径**（长文本）：滑动分片 → 保守归一化 → 代码内存合并（Jaccard 相似度）→ LLM 全局精修
- LRU 缓存 (16 条, 30 min TTL)，键含文本指纹防跨域污染

### Layer 2 — 分级判定

- 10 实体一批并行 LLM 调用
- 注入基础类型信息 + 域专属规则（从 DomainAdapter 读取）
- 失败批次自动拆为单实体重试
- LLM 不可用时回退基于基础类型的兜底规则

### Layer 3 — 交叉裁决

- LLM 全局冗余检测：降级决策空间重叠的实体，合并同义变体
- 代码层别名词典预合并减少 LLM 输入
- 文学/叙事域可配置跳过（`skip_layer3: true`）
- LRU 缓存 (64 条) + 方差日志持久化

### SemanticMediator — 7 大类基础类型

全领域统一中间层，所有实体先映射到通用基础类型：

| 基础类型 | 含义 | 最低 tier |
|---------|------|----------|
| **Agent** | 独立决策主体（国家/企业/政权/军阀） | tier1 |
| **Subordinate** | 附属参与者（官员/子公司/部门） | tier2 |
| **Resource** | 资源/工具/物资 | tier3 |
| **Geography** | 地理空间/城市/区域 | tier3 |
| **Contract** | 合约/条约/协议 | tier3 |
| **Event** | 事件/冲突/项目 | tier3 |
| **Concept** | 抽象概念/政策/数据 | tier3 |

### DomainAdapter — 配置驱动

所有领域专属规则外置为 `data/domain_adapters/xxx.yaml`，运行时动态加载。每个 YAML 包含：域元信息、参数阈值、基础类型映射表、tier 规则、Layer 配置、LLM Prompt 片段、别名词典和白名单。

**内置 12 个适配器**：`geo_strategy` `business` `military` `politics` `novel` `narrative` `history` `ecology` `urban` `tech` `info_war` `universal_neutral`

**新增领域**：只需新建一份 YAML 文件，无需修改任何 Python 代码。

### DomainDetector — 自动域探测

- 关键词特征匹配（从各适配器的 `detect_keywords` 聚合）
- 实体类型统计（Person / Organization / Company 占比）
- 置信度打分，最高分加载对应适配器
- 无匹配 → `universal_neutral` 通用中立兜底
- 手动传入 domain 参数优先

---

## 算法引擎

### ODE 微分方程

8 类方程：`decay` / `logistic` / `fatigue_recovery` / `supply_consumption` / `pollution_spread` / `resource_depletion` / `competitive_logistic` / `cash_flow_dynamics`。scipy RK45 自适应步长，降级 Euler 冻结态。

### 3D 物理引擎

刚体力学（重力、阻尼）、空间哈希碰撞检测（N > 150 时 O(N)）、各向同性高斯扩散、径向冲击波。

### 观点动力学（HK 模型）

Hegselmann-Krause 有界置信模型。`geo_strategy` ε = 0.07（最难动摇），`urban` ε = 0.25（最易相互影响）。

### 有限状态机（FSM）

每域专用状态循环。支持历史趋势条件、虚拟空间度量、对手指标感知、自动敌友划分、FSM override。非命令态 agent 由 FSM 确定性驱动，零 LLM 调用。

---

## 规则包体系

### 8 个内置领域

| 域 | 指标数 | 特性 |
|----|--------|------|
| ⚔️ military | 5 | 空间战斗、distance_to_enemy FSM |
| 📊 business | 6 | 品牌竞争、供应链保护 |
| 🏛️ politics | 6 | 支持率竞争、立法博弈 |
| 🔬 tech | 5 | tech_lead + talent_pool 双竞争 |
| 📰 info_war | 4 | public_trust + polarization 双竞争 |
| 🌿 ecology | 6 | 碳排放、污染治理 |
| 🏗️ urban | 7 | 土地规划、交通设施 |
| 🌐 geo_strategy | 28 | 7 域继承 + 跨域复合 + delay_effects |

### 规则包 JSON 结构

```json
{
  "domain_key": {
    "name": "显示名称",
    "inherited_from": ["ancestor"],
    "metrics": ["m1", "m2"],
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

支持用户通过 Web UI 上传自定义规则包 JSON。

---

## 策略优化器

蒙特卡洛多方案对比：
1. 用户定义多个候选策略方案
2. 每方案运行 N 次独立模拟（随机种子 + 温度抖动）
3. 量化模式 `RuleEngine.judge()` 客观判胜负
4. 统计分析：成功率 / 胜率 / CI95 / 成本
5. 帕累托前沿推荐最优方案

---

## 信息生态与人格演化

### 情报不对称

信任度 [-5, +5] 驱动情报延迟 [4, 0] 轮 + 失真 [0%, 30%]。

### 人格动态化

触发条件（共享闸门）：
- 环境漂移 > 5 或累计 > 12
- 盟友/对手集合变化
- 指标逼近淘汰线
- 超过 6 轮无反思

LLM 生成行为准则（≤20 字），写入 `system_prompt_extra`，FIFO 保留 3 条，注入后续决策。

---

## 存储架构

| 存储引擎 | 用途 |
|---------|------|
| **Kuzu** (嵌入式图数据库) | 实体节点、关系边、事件序列、因果归因 |
| **LanceDB** (向量数据库) | 语义分块索引、事件记忆（含可见性） |
| **SQLite** | 会话持久化、日志、Token 统计、报告 |

---

## 安装与运行

### 前置条件

- Python ≥ 3.11
- Node.js ≥ 20
- Rust 工具链（仅打包需要）

### 开发模式

```bash
pip install -e .
python run.py
# 访问 http://localhost:5173
```

### 打包安装包（Windows）

详见 `AGENTS.md` 完整打包流程。

---

## 配置参考

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FORGE_PROVIDER` | `lmstudio` | LLM 提供商 |
| `FORGE_LLM_BASE` | `http://127.0.0.1:1234/v1` | LLM API 地址 |
| `FORGE_LLM_MODEL` | — | LLM 模型名 |
| `FORGE_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | 嵌入模型 |
| `FORGE_MAX_CONCURRENT` | `2` | 并发 LLM 上限 |
| `FORGE_DEFAULT_ROUNDS` | `10` | 默认推演轮数 |
| `FORGE_RETRIEVE_TOP_K` | `5` | 语义检索返回数 |
| `FORGE_REPORT_MAX_TOKENS` | `30000` | 报告输出上限 |
| `FORGE_RULE_DIR` | — | 规则包目录（打包时用） |

---

## API 参考

所有端点前缀 `/api/forge`。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传源文档（txt/md/pdf/docx/json） |
| `/session` | POST | 创建会话 |
| `/sessions` | GET | 会话列表（全量，无限制） |
| `/session/{id}` | GET / DELETE | 会话详情 / 删除 |
| `/session/{id}/start` | POST | 启动推演 |
| `/session/{id}/pause` | POST | 暂停推演 |
| `/session/{id}/resume` | POST | 恢复推演 |
| `/session/{id}/start/cancel` | POST | 取消运行 |
| `/session/{id}/intervene` | POST | 用户干预注入 |
| `/session/{id}/fsm-override` | POST | FSM 强制动作 |
| `/session/{id}/pre-goal` | POST | 推演目标注入 |
| `/session/{id}/settings` | POST | 会话设置（多动作/天气/地形） |
| `/session/{id}/graph` | GET | 知识图谱数据 |
| `/session/{id}/timeline` | GET | 行动时序 |
| `/session/{id}/causal` | GET | 因果子图 |
| `/session/{id}/report` | GET | 推演报告 |
| `/session/{id}/logs` | GET | 推演日志 |
| `/session/{id}/tokens` | GET | Token 统计 |
| `/session/{id}/stream` | GET | SSE 事件流 |
| `/session/{id}/optimize` | POST | 策略优化 |
| `/session/{id}/optimize/result` | GET | 优化结果 |
| `/session/{id}/optimize/cancel` | POST | 取消优化 |
| `/domains` | GET | 可用领域列表 |
| `/rules/upload` | POST | 上传自定义规则包 |
| `/config/llm` | GET / POST | LLM 配置 |
| `/config/embedding` | GET / POST | 嵌入配置 |
| `/config/engine` | GET / POST | 引擎参数配置 |
| `/config/providers` | GET | 提供商列表 |
| `/config/list-models` | POST | 列出可用模型 |
| `/config/test-connection` | POST | 测试连接 |

---

## 项目结构

```
src/strategy_forge/
├── core/              配置、LLM 客户端、Token 计数、分块、规则模板
├── storage/           会话存储 (SQLite)、图谱存储 (Kuzu)
├── api/               FastAPI 路由
├── engine/            引擎核心
│   ├── orchestrator.py      五阶段流水线协调器
│   ├── entity_registry.py   实体注册中心（Layer 1/2/3 + SemanticMediator）
│   ├── domain_adapter.py    统一 YAML 领域适配器
│   ├── semantic_mediator.py 7 大类基础类型映射
│   ├── domain_detector.py   自动域探测
│   ├── agent_factory.py     智能体人设生成
│   ├── rule_engine.py       量化规则引擎
│   ├── simulator.py         模拟引擎
│   ├── reporter.py          报告生成
│   ├── optimizer.py         策略优化器
│   ├── preprocessor.py      语义分块 + 实体提取
│   ├── graph_builder.py     图谱构建
│   ├── seed_extractor.py    种子指标提取
│   ├── strategic_reasoner.py 战略推理
│   └── ...
└── algorithms/        算法模块
    ├── fsm_module.py        有限状态机
    ├── ode_module.py        ODE 微分方程
    ├── opinion_dynamics.py  观点动力学
    ├── physics_module.py    物理引擎
    └── pipeline_engine.py   算法链调度

data/
├── rule/              规则包 rules.json（8 域）
├── domain_adapters/   统一领域适配器 YAML（13 个域）
└── methodology.yaml   双模式通用方法论（角色推断/情绪/信任/反思）

apps/strategy-forge/   Tauri 桌面应用（React + TypeScript + Vite）
tests/                 测试
```

---

## 许可证

[Apache License 2.0](LICENSE)
