# StrategyForge 开发备忘

## 打包安装包流程（重要！）

StrategyForge 使用 **Tauri v2 + PyInstaller + NSIS** 打包桌面安装包。**绝对不允许用 zip/7z/自解压之类的野路子替代。**

### 步骤

```powershell
# 1. 构建后端 exe
python -m PyInstaller strategy-forge-backend.spec --noconfirm

# 2. 同步后端到 Tauri resources
$res = "apps\strategy-forge\src-tauri\resources\strategy-forge-backend"
Remove-Item "$res\_internal","$res\strategy-forge-backend.exe" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "dist\strategy-forge-backend\strategy-forge-backend.exe","dist\strategy-forge-backend\_internal" -Destination $res -Recurse -Force
Copy-Item "data\rule\rules.json" -Destination "$res\data\rule\rules.json" -Force
Copy-Item "data\rule\domain_prompts.json" -Destination "$res\data\rule\domain_prompts.json" -Force
Copy-Item "data\rule\entity_alias.json" -Destination "$res\data\rule\entity_alias.json" -Force
Copy-Item "data\rule\layer3_config.yaml" -Destination "$res\data\rule\layer3_config.yaml" -Force
Copy-Item "data\domain_adapters\*.yaml" -Destination "$res\data\domain_adapters\" -Force

# 3. 构建前端 + Tauri + NSIS 安装包
cd apps\strategy-forge
npx tauri build --bundles nsis

# 4. 复制安装包到 release 目录
Copy-Item "apps\strategy-forge\src-tauri\target\release\bundle\nsis\StrategyForge_*_x64-setup.exe" -Destination "release\StrategyForge_Setup.exe" -Force
```

### 架构

```
安装目录 (C:\Program Files\StrategyForge):
├── StrategyForge.exe              ← Tauri 原生壳 (系统托盘、后台常驻)
├── strategy-forge-backend\
│   ├── strategy-forge-backend.exe ← PyInstaller 后端 (FastAPI + uvicorn)
│   ├── _internal\                 ← Python 运行时依赖
│   └── data\rule\rules.json      ← 内置规则包
└── (前端内嵌在 Tauri WebView，不暴露文件)

运行期数据: %LOCALAPPDATA%\StrategyForge\data\  (Kuzu图数据库 + LanceDB向量库 + SQLite会话)
```

### 关键约定

- 前端 `API_BASE` 生产模式通过 `isTauri()` 检测切换：Tauri 用 `http://127.0.0.1:8000`，独立部署用相对路径 `/api/forge`
- 后端启动端口固定 `127.0.0.1:8000`
- 规则包路径：Tauri 通过 `FORGE_RULE_DIR` 指向安装目录，开发模式使用 `data/rule/`
- 不要修改 `src-tauri/src/main.rs` 中的端口/路径逻辑
- PyInstaller spec 中 `console=False`（GUI 模式，无黑窗）

### 测试模型

- 本地 LM Studio: `google/gemma-4-12b` (主推) / `qwen3.5-2b` (轻量验证)
- 环境变量: `FORGE_PROVIDER=lmstudio`, `FORGE_LLM_MODEL=...`
- 嵌入模型: `text-embedding-embeddinggemma-300m-qat`

## EntityRegistry 模块开发备忘

### 架构 (P2 重构后)

```
build_registry()
    ├── DomainDetector: 自动域探测 (手动优先)
    ├── DomainAdapter: 统一 YAML 配置加载 (data/domain_adapters/)
    ├── Layer 1: 实体归一化 (快速/分片双路径)
    ├── SemanticMediator: 7大类基础类型映射 (Agent/Subordinate/...)
    ├── Layer 2: 通用 tier 判定 (LLM，注入 adapter prompts)
    └── Layer 3: 交叉裁决 (跳过逻辑由 adapter.layer.skip_layer3 控制)
```

### 核心设计原则

- **零硬编码域分支**：所有 `if domain in ("novel", "history")` 等分支已全部消除
  - Layer3 跳过逻辑 → `adapter.layer.skip_layer3` (YAML 配置)
  - 方法论注入 → `adapter.meta.methodology_mode` (`geo`/`narrative`/`neutral`)
  - Jaccard 阈值 → `adapter.params.jaccard_threshold`
  - 采样策略 → `adapter.params.sampling` ({head_size, tail_size, lit_mode})
  - Token 限制 → `adapter.params.token` (每层独立配置)
  - L2/L3 prompts → `adapter.prompts` (YAML 注入)
  - 别名词典 → `adapter.aliases` (force_keep + org_members + person_country)

### 7 大类基础类型 (SemanticMediator)

全领域统一中间层，所有实体先映射到基础类型，再判定 tier：
- **Agent**: 独立决策主体 (国家/企业/政权/组织/NGO/军阀)
- **Subordinate**: 附属参与者 (官员/子公司/部门/人员)
- **Resource**: 资源/工具/物资/武器
- **Geography**: 地理空间/城市/区域
- **Contract**: 合约/条约/协议
- **Event**: 事件/冲突/项目
- **Concept**: 抽象概念/政策/数据指标

### 领域配置文件

统一配置格式 `data/domain_adapters/xxx.yaml`（由 `scripts/migrate_adapters.py` 从旧三套配置自动生成）：

| 文件 | 域 |
|------|-----|
| `geo_strategy.yaml` | 地缘战略 |
| `business.yaml` | 商业经济 |
| `military.yaml` | 军事战争 |
| `politics.yaml` | 政治博弈 |
| `novel.yaml` / `narrative.yaml` | 小说/叙事 |
| `history.yaml` | 历史 |
| `ecology.yaml` / `urban.yaml` / `tech.yaml` / `info_war.yaml` | 专项域 |
| `universal_neutral.yaml` | **未知域兜底** (零领域偏见) |

### 新增/修改域的流程

1. 在 `data/domain_adapters/` 下新增或修改 `xxx.yaml`
2. 无需修改任何 Python 代码
3. 自动生效：重启后端后 `DomainAdapter` 自动发现新文件

### 向前兼容

- `build_registry()` 入口签名不变 — 上游调用者零修改
- `EntityRegistry` / `RegisteredEntity` 数据结构不变
- 旧配置文件 `domain_prompts.json`、`entity_alias.json`、`layer3_config.yaml` 保留不删
- `RegisteredEntity` 新增可选字段 `base_type: str` (不影响现有序列化)
