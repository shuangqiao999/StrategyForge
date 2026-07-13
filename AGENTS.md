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
