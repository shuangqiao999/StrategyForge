import { useState, useEffect, useCallback, useRef } from "react";
import ForceGraph3D from "react-force-graph-3d";

const API_BASE = import.meta.env.DEV
  ? "/api/forge"
  : isTauri()
    ? "http://127.0.0.1:8000/api/forge"
    : "/api/forge";

function isTauri(): boolean {
  return !!(window as any).__TAURI_INTERNALS__;
}

const lbl: React.CSSProperties = { fontSize: 13, color: "#94a3b8", marginBottom: 4, display: "block" };
const inp: React.CSSProperties = { height: 32, marginBottom: 8, width: "100%" };
const btn: React.CSSProperties = { height: 28, fontSize: 13, borderRadius: 6, border: "1px solid #334155", cursor: "pointer", padding: "0 12px" };

// ── Types ──

interface SessionItem {
  id: string;
  title: string;
  status: string;
  phase: string;
  entity_count: number;
  relation_count: number;
  agent_count: number;
  current_round: number;
  total_rounds: number;
  created_at: string;
}

interface GraphData {
  nodes: Array<{ id: string; name: string; type: string; description: string }>;
  links: Array<{ source: string; target: string; relation: string; weight: number }>;
}

interface LogEntry {
  phase: string;
  message: string;
  timestamp: string;
}

interface TimelineAction { action: string; timestamp: string; description: string; event_type: string; effect?: string; driver?: string; }
interface AgentTimeline { agent_id: string; agent_name: string; actions: TimelineAction[]; }
interface TimelineData {
  timelines: AgentTimeline[];
  sequence: Array<{ timestamp: string; agent_name: string; action: string; description: string; event_type: string; effect?: string; driver?: string }>;
}

interface CausalNode { id: string; kind: string; label: string; desc?: string; }
interface CausalLink { source: string; target: string; type: string; label: string; }
interface CausalData {
  nodes: CausalNode[];
  links: CausalLink[];
  summary: Array<{ source: string; target: string; metric: string; amount: number }>;
}

interface ReportData {
  summary?: string;
  key_events?: Array<any>;
  risk_alerts?: string[];
  recommendations?: string[];
  quantified?: boolean;
  domain?: string;
  final_states?: Record<string, { name: string; metrics: Record<string, number>; history?: any[]; alive: boolean }>;
  causal_summary?: string[];
  stage_narratives?: Array<{ stage: string; round_range: string; start_state: string; key_decisions: string; causal_logic: string; end_state: string }>;
  deviation_analysis?: Array<{ round: number; agent: string; decision: string; deviation_level: string; reason: string }>;
  conclusion?: string;
}

interface TokenStats {
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  phases: Record<string, { prompt: number; completion: number; total: number }>;
  rounds: Record<string, { prompt: number; completion: number; total: number }>;
}

// ── Phase Labels ──

const PHASE_LABELS: Record<string, string> = {
  created: "已创建",
  ontology_running: "本体生成中...",
  graph_running: "图谱构建中...",
  agents_running: "智能体生成中...",
  simulating: "模拟推演中...",
  reporting: "报告生成中...",
  optimizing: "策略优化中...",
  complete: "已完成",
  failed: "失败",
  paused: "已暂停",
};
const RUNNING_SET = new Set(["ontology_running","graph_running","agents_running","simulating","reporting","optimizing"]);

const METRIC_LABELS: Record<string, string> = {
  strength: "军力", morale: "士气", supply: "补给", fatigue: "疲劳度", leadership: "领导力",
  market_share: "市场份额", cash_flow: "现金流", brand: "品牌", rnd: "研发",
  supply_chain: "供应链", support_rate: "支持率", economy: "经济", unity: "团结度",
  intl_relations: "国际关系", legislative_power: "立法权",
  population: "人口", resources: "资源", pollution: "污染", biodiversity: "生物多样性",
  stability: "稳定性", employment: "就业", infrastructure: "基础设施", finance: "财政",
  satisfaction: "满意度", tech_lead: "技术领先", chip_stock: "芯片储备",
  talent_pool: "人才池", patent_barrier: "专利壁垒", commercialization: "商业化",
  narrative_dominance: "舆情主导", public_trust: "公信力", polarization: "极化度",
  media_reach: "媒体触达",
};
const fmtMetric = (m: string) => METRIC_LABELS[m] || m;

const Toggle = ({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) => (
  <label style={{ display: "inline-flex", alignItems: "center", flexShrink: 0, cursor: "pointer" }}>
    <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)}
      style={{ position: "absolute", opacity: 0, width: 0, height: 0 }} />
    <span style={{
      position: "relative", display: "inline-block", width: 36, height: 20,
      borderRadius: 10, background: checked ? "#22c55e" : "#475569",
      transition: "background 0.2s ease",
    }}>
      <span style={{
        position: "absolute", top: 2, left: checked ? 18 : 2,
        width: 16, height: 16, borderRadius: "50%", background: "#fff",
        transition: "left 0.2s ease", boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
      }} />
    </span>
  </label>
);

// ── Main App ──

export default function App() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [sourceMaterial, setSourceMaterial] = useState("");
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [report, setReport] = useState<ReportData | null>(null);
  const [timeline, setTimeline] = useState<TimelineData | null>(null);
  const [causal, setCausal] = useState<CausalData | null>(null);
  const [tokenData, setTokenData] = useState<TokenStats | null>(null);
  const [timelineView, setTimelineView] = useState<"timeline" | "causal">("timeline");
  const [mainTab, setMainTab] = useState<"graph" | "report" | "logs" | "timeline" | "optimize" | "token" | "dashboard">("graph");
  const [domain, setDomain] = useState("auto");
  const [rounds, setRounds] = useState(10);
  const [roundMode, setRoundMode] = useState<"preset" | "custom">("preset");
  const [domains, setDomains] = useState<Array<{domain:string;name:string}>>([]);

  // ── 策略优化器 ──
  const [optEnabled, setOptEnabled] = useState(false);
  const [optIterations, setOptIterations] = useState(20);
  const [optObjective, setOptObjective] = useState("balanced");
  const [optMultiAction, setOptMultiAction] = useState(false);
  const [optMaxActions, setOptMaxActions] = useState(3);
  const [optWinCondition, setOptWinCondition] = useState("");
  const [optScenarios, setOptScenarios] = useState<Array<{ name: string; directive: string; entity_ref: string }>>([{ name: "方案 1", directive: "", entity_ref: "" }]);
  const [optRunning, setOptRunning] = useState(false);
  const [optProgress, setOptProgress] = useState<{ done: number; total: number; current: string; best_win: number } | null>(null);
  const [optReport, setOptReport] = useState<any>(null);
  const optPollRef = useRef<number | null>(null);
  const [selectedCausalNode, setSelectedCausalNode] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [preGoal, setPreGoal] = useState("");
  const [interventionText, setInterventionText] = useState("");
  const [sending, setSending] = useState(false);
  const [ovAgent, setOvAgent] = useState<string | null>(null);
  const [ovAction, setOvAction] = useState("");
  const [ovIntensity, setOvIntensity] = useState(0.6);
  const [ovTarget, setOvTarget] = useState("");
  const [ovRounds, setOvRounds] = useState(1);
  const logsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [trendSnapshots, setTrendSnapshots] = useState<Array<{round: number, entities: Array<{name:string, metrics:Record<string,number>, alive:boolean}>}>>([]);
  const graphRef = useRef<any>(null);
  const causalGraphRef = useRef<any>(null);
  const sseSidRef = useRef<string | null>(null);

  const zoomGraph = (rf: React.RefObject<any>, factor: number) => {
    const fg = rf.current; if (!fg) return;
    const pos = fg.cameraPosition();
    fg.cameraPosition({ x: pos.x * factor, y: pos.y * factor, z: pos.z * factor }, { x: 0, y: 0, z: 0 } as any, 300);
  };
  const resetGraph = (rf: React.RefObject<any>) => { rf.current?.zoomToFit(400, 50); };

  // ── Settings ──
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"llm" | "embed" | "engine">("llm");
  const [cfgLLMBase, setCfgLLMBase] = useState("");
  const [cfgLLMKey, setCfgLLMKey] = useState("");
  const [cfgLLMModel, setCfgLLMModel] = useState("");
  const [cfgLLMProvider, setCfgLLMProvider] = useState("");
  const [cfgLLMTemp, setCfgLLMTemp] = useState(0.3);
  const [cfgEmbedBase, setCfgEmbedBase] = useState("");
  const [cfgEmbedKey, setCfgEmbedKey] = useState("");
  const [cfgEmbedModel, setCfgEmbedModel] = useState("");
  const [cfgEmbedProvider, setCfgEmbedProvider] = useState("");
  const [cfgFetchingModels, setCfgFetchingModels] = useState(false);
  const [cfgLLMFetchedModels, setCfgLLMFetchedModels] = useState<string[]>([]);
  const [cfgLLMModelError, setCfgLLMModelError] = useState("");
  const [cfgEmbedFetchedModels, setCfgEmbedFetchedModels] = useState<string[]>([]);
  const [cfgEmbedModelError, setCfgEmbedModelError] = useState("");
  const [cfgSaving, setCfgSaving] = useState(false);
  const [cfgLLMTest, setCfgLLMTest] = useState<"" | "testing" | "ok" | "fail">("");
  const [cfgProviders, setCfgProviders] = useState<Array<{slug:string;name:string;default_llm_base_url:string;default_llm_model:string;default_embed_model:string;note:string}>>([]);

  // ── 引擎配置 ──
  const [cfgDefaultRounds, setCfgDefaultRounds] = useState(10);
  const [cfgMaxAgents, setCfgMaxAgents] = useState(10000);
  const [cfgCandidateCount, setCfgCandidateCount] = useState(3);
  const [cfgMaxConcurrent, setCfgMaxConcurrent] = useState(2);
  const [cfgRetrieveTopK, setCfgRetrieveTopK] = useState(5);
  const [cfgSimilarity, setCfgSimilarity] = useState(0.4);
  const [cfgSafetyNet, setCfgSafetyNet] = useState(true);
  const [cfgRecallBoost, setCfgRecallBoost] = useState(true);
  const [cfgEventHybrid, setCfgEventHybrid] = useState(true);
  const [cfgLLMTimeout, setCfgLLMTimeout] = useState(300);
  const [cfgConnectTimeout, setCfgConnectTimeout] = useState(60);
  const [cfgGenTimeout, setCfgGenTimeout] = useState(1800);
  const [cfgRetryPasses, setCfgRetryPasses] = useState(3);
  const [cfgFailThreshold, setCfgFailThreshold] = useState(0.75);

  // ── 模型类型甄别：嵌入模型关键词 ──
  const EMBED_MODEL_KW = ["embed", "embedding", "bge", "e5", "gte", "stella", "nomic", "jina"];

  const fetchConfig = useCallback(async (): Promise<boolean> => {
    try {
      const [lr, er, pr] = await Promise.all([
        fetch(`${API_BASE}/config/llm`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/config/embedding`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/config/providers`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/config/engine`).then(r => r.ok ? r.json() : null),
      ]);
      if (lr) { setCfgLLMBase(lr.llm_base_url || ""); setCfgLLMKey(lr.llm_api_key || ""); setCfgLLMModel(lr.llm_model || ""); setCfgLLMProvider(lr.provider_slug || ""); setCfgLLMTemp(lr.llm_temperature || 0.3); }
      if (er) { setCfgEmbedBase(er.embedding_api_base || ""); setCfgEmbedKey(er.embedding_api_key || ""); setCfgEmbedModel(er.embedding_model_name || ""); setCfgEmbedProvider(er.provider_slug || ""); }
      if (pr) { setCfgProviders(pr.providers || []); }
      const eng = await fetch(`${API_BASE}/config/engine`).then(r => r.ok ? r.json() : null);
      if (eng) {
        setCfgDefaultRounds(eng.default_rounds ?? 10); setCfgMaxAgents(eng.max_agents ?? 10000);
        setCfgCandidateCount(eng.candidate_count ?? 3); setCfgMaxConcurrent(eng.max_concurrent ?? 2);
        setCfgRetrieveTopK(eng.retrieve_top_k ?? 5); setCfgSimilarity(eng.similarity_threshold ?? 0.4);
        setCfgSafetyNet(eng.intel_safety_net ?? true); setCfgRecallBoost(eng.recall_rel_boost ?? true);
        setCfgEventHybrid(eng.event_hybrid ?? true); setCfgLLMTimeout(eng.llm_timeout ?? 300);
        setCfgConnectTimeout(eng.connect_timeout ?? 60); setCfgGenTimeout(eng.generation_timeout ?? 1800);
        setCfgRetryPasses(eng.retry_passes ?? 3); setCfgFailThreshold(eng.sim_fail_threshold ?? 0.75);
      }
      return !!(pr && (pr.providers || []).length);
    } catch { return false; }
  }, []);

  // 后端 exe 由 Tauri 启动需数秒就绪，挂载时轮询重试直到服务商加载成功。
  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const tick = async () => {
      if (cancelled) return;
      const ok = await fetchConfig();
      attempts += 1;
      if (!ok && attempts < 20 && !cancelled) setTimeout(tick, 1500);
    };
    tick();
    return () => { cancelled = true; };
  }, [fetchConfig]);

  // 动态加载规则包领域列表（内置 + 自定义），解耦前端硬编码
  useEffect(() => { fetchDomains(); }, []);

  const fetchModels = useCallback(async () => {
    const base = settingsTab === "llm" ? cfgLLMBase : cfgEmbedBase;
    const key = settingsTab === "llm" ? cfgLLMKey : cfgEmbedKey;
    if (!base) return;
    setCfgFetchingModels(true);
    if (settingsTab === "llm") { setCfgLLMModelError(""); setCfgLLMFetchedModels([]); }
    else { setCfgEmbedModelError(""); setCfgEmbedFetchedModels([]); }
    try {
      const r = await fetch(`${API_BASE}/config/list-models`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_type: "openai", base_url: base, api_key: key }),
      });
      const data = await r.json();
      const all: string[] = data.models || [];
      if (data.error) {
        if (settingsTab === "llm") setCfgLLMModelError(data.error);
        else setCfgEmbedModelError(data.error);
      } else if (settingsTab === "llm") {
        // LLM tab: 过滤掉嵌入模型
        const filtered = all.filter(m => !EMBED_MODEL_KW.some(k => m.toLowerCase().includes(k)));
        setCfgLLMFetchedModels(filtered);
        if (filtered.length === 0) setCfgLLMModelError("未检测到对话模型（已过滤嵌入模型），请检查模型列表");
      } else {
        // Embedding tab: 仅保留嵌入模型，若全过滤则留全部
        const filtered = all.filter(m => EMBED_MODEL_KW.some(k => m.toLowerCase().includes(k)));
        setCfgEmbedFetchedModels(filtered.length > 0 ? filtered : all);
      }
    } catch (e: any) {
      if (settingsTab === "llm") setCfgLLMModelError(e.message || "Failed");
      else setCfgEmbedModelError(e.message || "Failed");
    }
    setCfgFetchingModels(false);
  }, [settingsTab, cfgLLMBase, cfgLLMKey, cfgEmbedBase, cfgEmbedKey]);

  const testLLM = useCallback(async () => {
    setCfgLLMTest("testing");
    try {
      const r = await fetch(`${API_BASE}/config/test-connection`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_type: "openai", base_url: cfgLLMBase, api_key: cfgLLMKey }),
      });
      const data = await r.json();
      setCfgLLMTest(data.ok ? "ok" : "fail");
    } catch { setCfgLLMTest("fail"); }
  }, [cfgLLMBase, cfgLLMKey]);

  const saveConfig = useCallback(async () => {
    setCfgSaving(true);
    try {
      await fetch(`${API_BASE}/config/llm`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_base_url: cfgLLMBase, llm_api_key: cfgLLMKey, llm_model: cfgLLMModel, provider_slug: cfgLLMProvider, llm_temperature: cfgLLMTemp }),
      });
      await fetch(`${API_BASE}/config/embedding`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ embedding_api_base: cfgEmbedBase, embedding_api_key: cfgEmbedKey, embedding_model_name: cfgEmbedModel, provider_slug: cfgEmbedProvider }),
      });
      await fetch(`${API_BASE}/config/engine`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          default_rounds: cfgDefaultRounds, max_agents: cfgMaxAgents, candidate_count: cfgCandidateCount,
          llm_temperature: cfgLLMTemp, max_concurrent: cfgMaxConcurrent,
          retrieve_top_k: cfgRetrieveTopK, similarity_threshold: cfgSimilarity,
          intel_safety_net: cfgSafetyNet, recall_rel_boost: cfgRecallBoost, event_hybrid: cfgEventHybrid,
          llm_timeout: cfgLLMTimeout, connect_timeout: cfgConnectTimeout, generation_timeout: cfgGenTimeout,
          retry_passes: cfgRetryPasses, sim_fail_threshold: cfgFailThreshold,
        }),
      });
      await fetchConfig();
      setShowSettings(false);
    } catch { /* ignore */ }
    setCfgSaving(false);
  }, [cfgLLMBase, cfgLLMKey, cfgLLMModel, cfgLLMProvider, cfgLLMTemp, cfgEmbedBase, cfgEmbedKey, cfgEmbedModel, cfgEmbedProvider,
      cfgDefaultRounds, cfgMaxAgents, cfgCandidateCount, cfgMaxConcurrent, cfgRetrieveTopK, cfgSimilarity,
      cfgSafetyNet, cfgRecallBoost, cfgEventHybrid, cfgLLMTimeout, cfgConnectTimeout, cfgGenTimeout,
      cfgRetryPasses, cfgFailThreshold, fetchConfig]);

  const fetchSessions = useCallback(async (): Promise<boolean> => {
    try {
      const r = await fetch(`${API_BASE}/sessions`);
      if (r.ok) { setSessions(await r.json()); return true; }
      return false;
    } catch { return false; }
  }, []);

  // 冷启动时后端可能尚未就绪，轮询重试直到会话列表加载成功。
  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const tick = async () => {
      if (cancelled) return;
      const ok = await fetchSessions();
      attempts += 1;
      if (!ok && attempts < 20 && !cancelled) setTimeout(tick, 1500);
    };
    tick();
    return () => { cancelled = true; };
  }, [fetchSessions]);

  const fetchGraph = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/graph`);
      if (r.ok) setGraphData(await r.json());
    } catch { /* ignore */ }
  }, []);

  const fetchTimeline = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/timeline`);
      if (r.ok) setTimeline(await r.json());
    } catch { /* ignore */ }
  }, []);

  const fetchCausal = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/causal`);
      if (r.ok) setCausal(await r.json());
    } catch { /* ignore */ }
  }, []);

  const fetchTokens = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/tokens`);
      if (r.ok) {
        const d = await r.json();
        setTokenData(d.stats && Object.keys(d.stats).length > 0 ? d.stats : null);
      }
    } catch (e: any) { console.error("Token fetch failed:", e.message); }
  }, []);

  const fetchDomains = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/domains`);
      if (r.ok) setDomains((await r.json()).domains || []);
    } catch { /* ignore */ }
  }, []);

  // 运行前把多动作设置写入会话 config_json（普通推演与优化器统一读取）
  const persistSettings = useCallback(async (sessionId: string) => {
    try {
      await fetch(`${API_BASE}/session/${sessionId}/settings`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enable_multi_action: optMultiAction, max_actions: optMaxActions }),
      });
    } catch { /* ignore */ }
  }, [optMultiAction, optMaxActions]);

  const fetchLogs = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/logs`);
      if (r.ok) setLogs(await r.json());
    } catch { /* ignore */ }
  }, []);

  const fetchReport = useCallback(async (sessionId: string) => {
    try {
      const r = await fetch(`${API_BASE}/session/${sessionId}/report`);
      if (r.ok) {
        const d = await r.json();
        const rep = d.report || null;
        setReport(rep && Object.keys(rep).length ? rep : null);
      } else setReport(null);
    } catch { setReport(null); }
  }, []);

  const selectSession = useCallback((id: string) => {
    setSelectedId(id);
    setMainTab("graph");
    fetchGraph(id);
    fetchLogs(id);
    fetchReport(id);
    fetchTimeline(id);
    fetchCausal(id);
    fetchTokens(id);
  }, [fetchGraph, fetchLogs, fetchReport, fetchTimeline, fetchCausal]);

  const handleCreate = useCallback(async () => {
    if (!title.trim() || !sourceMaterial.trim()) return;
    setCreating(true);
    try {
      const r = await fetch(`${API_BASE}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, source_material: sourceMaterial, config: { domain, total_rounds: rounds } }),
      });
      if (r.ok) {
        const data = await r.json();
        setSelectedId(data.id);
        setSessions(prev => [{ id: data.id, title: data.title, status: data.status, phase: "", entity_count: 0, relation_count: 0, agent_count: 0, current_round: 0, total_rounds: rounds, created_at: data.created_at }, ...prev]);
      }
    } catch { /* ignore */ }
    setCreating(false);
  }, [title, sourceMaterial, domain, rounds]);

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split(".").pop()?.toLowerCase();
    const allowed = ["txt","md","json","pdf","docx","py","js","ts","rs","go","java","c","cpp","h","csv","log","yaml","yml"];
    if (!ext || !allowed.includes(ext)) {
      e.target.value = "";
      return;
    }
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
      if (!r.ok) { const err = await r.text(); throw new Error(err); }
      const data = await r.json();
      setSourceMaterial(data.text_content);
      const titleHint = file.name.replace(/\.[^.]+$/, "").slice(0, 40);
      if (!title.trim()) setTitle(titleHint);
    } catch (err: any) {
      alert("文件上传失败: " + (err.message || "未知错误"));
    }
    setUploading(false);
    e.target.value = "";
  }, [title]);

  const handleStart = useCallback(async () => {
    if (!selectedId) return;
    setLoading(true);
    setLogs([]);
    try {
      await persistSettings(selectedId);
      const r = await fetch(`${API_BASE}/session/${selectedId}/start`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      await fetchSessions();
    } catch (e: any) {
      alert("推演启动失败: " + (e.message || "未知错误"));
    }
    setLoading(false);
  }, [selectedId, fetchSessions, persistSettings]);

  const handleCancel = useCallback(async () => {
    if (!selectedId) return;
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/session/${selectedId}/pause`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
    } catch (e: any) {
      alert("停止推演失败: " + (e.message || "未知错误"));
    }
    setLoading(false);
    await fetchSessions();
  }, [selectedId, fetchSessions]);

  const handleResume = useCallback(async () => {
    if (!selectedId) return;
    setLoading(true);
    setLogs([]);
    try {
      await persistSettings(selectedId);
      const r = await fetch(`${API_BASE}/session/${selectedId}/resume`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      setLoading(false);
      await fetchSessions();
    } catch (e: any) {
      setLoading(false);
      alert("继续推演失败: " + (e.message || "未知错误"));
    }
    setLoading(false);
  }, [selectedId, fetchSessions, persistSettings]);

  const handleDelete = useCallback(async (id: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (!window.confirm("确定删除该推演记录？将同时清除图谱、向量库与会话数据，且不可恢复。")) return;
    try {
      const r = await fetch(`${API_BASE}/session/${id}`, { method: "DELETE" });
      if (!r.ok) throw new Error(await r.text());
      if (selectedId === id) { setSelectedId(null); setGraphData(null); setLogs([]); setReport(null); }
      fetchSessions();
    } catch (err: any) {
      alert("删除失败: " + (err.message || "未知错误"));
    }
  }, [selectedId, fetchSessions]);

  // ── 策略优化器函数 ──
  const addScenario = useCallback(() => {
    setOptScenarios(prev => [...prev, { name: `方案 ${prev.length + 1}`, directive: "", entity_ref: "" }]);
  }, []);
  const removeScenario = useCallback((idx: number) => {
    setOptScenarios(prev => prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx));
  }, []);
  const updateScenario = useCallback((idx: number, field: "name" | "directive" | "entity_ref", val: string) => {
    setOptScenarios(prev => prev.map((s, i) => i === idx ? { ...s, [field]: val } : s));
  }, []);

  const pollOptimize = useCallback((id: string) => {
    if (optPollRef.current) window.clearInterval(optPollRef.current);
    optPollRef.current = window.setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/session/${id}/optimize/result`);
        if (!r.ok) return;
        const d = await r.json();
        setOptProgress(d.progress || null);
        if (d.report && Object.keys(d.report).length) setOptReport(d.report);
        if (!d.running) {
          if (optPollRef.current) { window.clearInterval(optPollRef.current); optPollRef.current = null; }
          setOptRunning(false);
          fetchSessions();
          fetchLogs(id);
        }
      } catch { /* ignore */ }
    }, 2000);
  }, [fetchSessions, fetchLogs]);

  const startOptimize = useCallback(async () => {
    if (!selectedId) return;
    const scenarios = optScenarios.filter(s => s.directive.trim()).map(s => ({
      name: s.name, directive: s.directive,
      win_target: s.entity_ref.trim() ? { entity_ref: s.entity_ref.trim() } : undefined,
    }));
    if (scenarios.length === 0) { alert("请至少填写一个方案的策略指令"); return; }
    if (!optWinCondition.trim() && !window.confirm("未填写胜利条件，将尝试使用会话的推演前目标(pre-goal)。是否继续？")) return;
    setOptRunning(true); setOptReport(null); setOptProgress(null); setMainTab("optimize"); setLogs([]);
    try {
      await persistSettings(selectedId);
      const r = await fetch(`${API_BASE}/session/${selectedId}/optimize`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenarios, win_condition: optWinCondition, iterations: optIterations, objective: optObjective }),
      });
      if (!r.ok) throw new Error(await r.text());
      pollOptimize(selectedId);
    } catch (e: any) {
      setOptRunning(false);
      alert("优化启动失败: " + (e.message || "未知错误"));
    }
  }, [selectedId, optScenarios, optWinCondition, optIterations, optObjective, persistSettings, pollOptimize]);

  const cancelOptimize = useCallback(async () => {
    if (!selectedId) return;
    try { await fetch(`${API_BASE}/session/${selectedId}/optimize/cancel`, { method: "POST" }); } catch { /* ignore */ }
  }, [selectedId]);

  const sendPreGoal = useCallback(async () => {
    if (!selectedId || !preGoal.trim()) return;
    try {
      await fetch(`${API_BASE}/session/${selectedId}/pre-goal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: preGoal }),
      });
      setPreGoal("");
    } catch { /* ignore */ }
  }, [selectedId, preGoal]);

  const sendIntervention = useCallback(async () => {
    if (!selectedId || !interventionText.trim()) return;
    setSending(true);
    try {
      await fetch(`${API_BASE}/session/${selectedId}/intervene`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: interventionText, scope: "during" }),
      });
      setInterventionText("");
      await fetchLogs(selectedId);
    } catch (err: any) {
      alert("干预发送失败: " + (err.message || "未知错误"));
    }
    setSending(false);
  }, [selectedId, interventionText, fetchLogs]);

  const submitFsmOverride = useCallback(async (agent: string) => {
    if (!selectedId || !agent || !ovAction.trim()) { alert("请填写强制动作"); return; }
    try {
      const r = await fetch(`${API_BASE}/session/${selectedId}/fsm-override`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent, action_type: ovAction.trim(), intensity: ovIntensity, target: ovTarget.trim(), rounds: ovRounds }),
      });
      if (r.ok) { setOvAgent(null); setOvAction(""); setOvTarget(""); await fetchLogs(selectedId); }
      else alert("强制失败: " + (await r.text()));
    } catch (err: any) {
      alert("强制发送失败: " + (err.message || "未知错误"));
    }
  }, [selectedId, ovAction, ovIntensity, ovTarget, ovRounds, fetchLogs]);

  // SSE auto-refresh logs, graph, timeline during ALL running phases
  useEffect(() => {
    if (!selectedId) return;
    const selected = sessions.find(s => s.id === selectedId);
    if (!selected) return;
    const runningSet = new Set(["ontology_running","graph_running","agents_running","simulating","reporting","optimizing"]);
    if (!runningSet.has(selected.status)) { sseSidRef.current = null; return; }
    if (sseSidRef.current === selectedId) return; // already connected
    sseSidRef.current = selectedId;
    const es = new EventSource(`${API_BASE}/session/${selectedId}/stream`);
    es.onmessage = (ev: MessageEvent) => {
      if (ev.data === "[DONE]") { es.close(); sseSidRef.current = null; fetchSessions(); fetchGraph(selectedId); fetchReport(selectedId); fetchTimeline(selectedId); fetchCausal(selectedId); fetchTokens(selectedId); return; }
      try {
        const d = JSON.parse(ev.data);
        if (d.type === "round") {
          if (d.snapshot) {
            setSnapshot(d.snapshot);
            if (d.snapshot.entities && d.round) {
              setTrendSnapshots(prev => {
                const filtered = prev.filter(s => s.round !== d.round);
                return [...filtered, {round: d.round, entities: d.snapshot.entities}].sort((a,b) => a.round - b.round);
              });
            }
          }
          fetchGraph(selectedId);
          fetchTimeline(selectedId);
          fetchCausal(selectedId);
          fetchTokens(selectedId);
        } else if (d.type === "status") {
          if (["complete","failed"].includes(d.status)) return; // [DONE] will handle terminal status
          fetchSessions();
        } else if (d.type === "error") {
          // ignore
        } else {
                setLogs(prev => [...prev, { phase: d.phase || "", message: d.message || "", timestamp: d.timestamp || "" }]);
          if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
        }
      } catch { /* ignore */ }
    };
    es.onerror = () => { es.close(); sseSidRef.current = null; fetchSessions(); };
    return () => { if (sseSidRef.current !== selectedId) es.close(); };  // do NOT clear sseSidRef here; session still connected
  }, [selectedId, sessions]);

  // Token 统计：前 3 次每 10 秒拉取，之后每 2 分钟
  useEffect(() => {
    if (!selectedId) return;
    let count = 0;
    let timer: number;
    const runningSet = new Set(["ontology_running","graph_running","agents_running","simulating","reporting","optimizing"]);
    const tick = () => {
      const sel = sessions.find(s => s.id === selectedId);
      if (sel && runningSet.has(sel.status)) {
        fetchTokens(selectedId);
      }
      count++;
      const delay = count < 3 ? 10000 : 120000;
      timer = window.setTimeout(tick, delay);
    };
    tick();
    return () => window.clearTimeout(timer);
  }, [selectedId, sessions, fetchTokens]);

  // 切换到 Token tab 时立即拉取
  useEffect(() => {
    if (mainTab === "token" && selectedId) {
      fetchTokens(selectedId);
    }
  }, [mainTab, selectedId, fetchTokens]);

  const selected = sessions.find(s => s.id === selectedId);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", height: "100%", overflow: "hidden" }}>
      {/* ── Left Panel: Sessions ── */}
      <div style={{ borderRight: "1px solid #374151", overflow: "auto", padding: 12 }}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>StrategyForge 战略推演</span>
          <button onClick={() => { fetchConfig(); setShowSettings(true); }} style={{ background: "#334155", border: "1px solid #475569", color: "#e2e8f0", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontSize: 12 }}>⚙ 配置</button>
        </h3>

        <div className="card" style={{ marginBottom: 10 }}>
          <input
            style={{ height: 32, marginBottom: 6, width: "100%" }}
            placeholder="会话标题"
            value={title}
            onChange={e => setTitle(e.target.value)}
          />
          <textarea
            style={{ height: 100, fontSize: 13, marginBottom: 6, width: "100%" }}
            placeholder="粘贴种子材料（或点击上传文档）"
            value={sourceMaterial}
            onChange={e => setSourceMaterial(e.target.value)}
          />
          <textarea
            style={{ height: 48, fontSize: 13, marginBottom: 6, width: "100%" }}
            placeholder="推演前愿景/目标（可选）"
            value={preGoal}
            onChange={e => setPreGoal(e.target.value)}
          />
          <div style={{ display: "flex", gap: 6, marginBottom: 6, alignItems: "center" }}>
            <select
              value={roundMode === "preset" ? String(rounds) : "custom"}
              onChange={e => {
                const v = e.target.value;
                if (v === "custom") { setRoundMode("custom"); return; }
                setRoundMode("preset");
                setRounds(parseInt(v));
              }}
              title="推演轮数：每轮所有智能体做出一次决策并执行一轮世界演化（ODE/FSM/Physics）"
              style={{ height: 32, flex: 1, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6, fontSize: 13 }}
            >
              <option value="5">{">"} 5 轮</option>
              <option value="10">{">"} 10 轮（默认）</option>
              <option value="15">{">"} 15 轮</option>
              <option value="20">{">"} 20 轮</option>
              <option value="30">{">"} 30 轮</option>
              <option value="50">{">"} 50 轮</option>
              <option value="100">{">"} 100 轮</option>
              <option value="custom">✎ 手动输入...</option>
            </select>
            {roundMode === "custom" && (
              <input
                type="number"
                min={1} max={100}
                value={rounds}
                onChange={e => { const v = parseInt(e.target.value); if (v >= 1 && v <= 100) setRounds(v); }}
                style={{ width: 70, height: 32, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6, fontSize: 13, textAlign: "center" }}
                title="手动输入 1-100 轮"
              />
            )}
          </div>
          <select
            value={domain}
            onChange={e => setDomain(e.target.value)}
            title="推演领域：选具体领域或自动识别进入量化推演；纯叙事保持 1.0 行为"
            style={{ height: 32, marginBottom: 6, width: "100%", background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6, fontSize: 13 }}
          >
            <option value="auto">🤖 自动识别领域（量化）</option>
            {domains.length === 0
              ? <option disabled value="">⚠️ 请上传规则包 JSON（无内置规则包可用）</option>
              : domains.map(d => <option key={d.domain} value={d.domain}>{d.name}</option>)
            }
            <option value="narrative">📖 纯叙事（不量化）</option>
          </select>
          {domain !== "narrative" && (
            <div style={{ marginBottom: 6 }}>
              <input type="file" accept=".json" style={{ display: "none" }} onChange={async e => {
                const f = e.target.files?.[0]; if (!f) return;
                const text = await f.text();
                try { JSON.parse(text); } catch { alert("无效 JSON 文件"); return; }
                const r = await fetch(`${API_BASE}/rules/upload`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ domain: f.name.replace(".json",""), content: text }) });
                if (r.ok) { await fetchDomains(); alert("规则包已上传并加载"); } else { alert("上传失败: " + (await r.text())); }
                e.target.value = "";
              }} id="rules-upload" />
              <button
                style={{ width: "100%", height: 30, fontSize: 12, background: "#1e293b", border: "1px solid #374151", borderRadius: 6, cursor: "pointer", color: "#cbd5e1", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}
                onClick={() => (document.getElementById("rules-upload") as HTMLInputElement)?.click()}
              >📤 上传自定义规则包</button>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.json,.pdf,.docx,.py,.js,.ts,.rs,.go,.java,.c,.cpp,.csv,.log,.yaml,.yml"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          <button
            style={{ width: "100%", height: 28, fontSize: 13, marginBottom: 6, background: "#1e293b", border: "1px solid #374151", borderRadius: 6, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, color: "#94a3b8" }}
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? "⏳" : "📎"}{" "}
            {uploading ? "解析中..." : "上传文档"}
          </button>
          <button
            className="btnPrimary"
            style={{ width: "100%", height: 32, fontSize: 13 }}
            onClick={handleCreate}
            disabled={creating}
          >
            {creating ? "创建中..." : "创建推演会话"}
          </button>
        </div>

        {/* ── 策略优化器面板（高级，默认关闭） ── */}
        <div style={{ marginBottom: 10, background: "#0f172a", border: "1px solid #334155", borderRadius: 8, padding: 10 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13, color: "#e2e8f0" }}>
            <Toggle checked={optEnabled} onChange={setOptEnabled} />
            策略优化器
            <span style={{ fontSize: 11, color: "#94a3b8", background: "#1e293b", borderRadius: 8, padding: "1px 6px" }}>Beta</span>
            <span style={{ marginLeft: "auto", fontSize: 12, color: optEnabled ? "#34d399" : "#64748b" }}>{optEnabled ? "已启用" : "默认关闭"}</span>
          </label>
          {optEnabled && (
            <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
              <div>
                <label style={lbl}>每方案模拟次数：{optIterations}</label>
                <input type="range" min={2} max={100} step={1} value={optIterations} onChange={e => setOptIterations(parseInt(e.target.value))} style={{ width: "100%" }} />
              </div>
              <div>
                <label style={lbl}>优化目标</label>
                <select value={optObjective} onChange={e => setOptObjective(e.target.value)} style={{ ...inp, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6 }}>
                  <option value="max_win_rate">📈 最高胜率</option>
                  <option value="min_cost">💰 最低成本/风险</option>
                  <option value="balanced">⚖️ 平衡（帕累托最优）</option>
                </select>
              </div>
              <div>
                <label style={lbl}>胜利条件（统一判定标准）</label>
                <textarea value={optWinCondition} onChange={e => setOptWinCondition(e.target.value)} placeholder="例：核心势力长期存续且主要人物善终（留空则用会话的推演前目标）" style={{ height: 50, fontSize: 13, width: "100%" }} />
              </div>
              <div>
                <label style={lbl}>候选方案（不同战略指令，逐一对比）</label>
                {optScenarios.map((s, i) => (
                  <div key={i} style={{ marginBottom: 6, background: "#1e293b", borderRadius: 6, padding: 6 }}>
                    <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
                      <input value={s.name} onChange={e => updateScenario(i, "name", e.target.value)} placeholder="方案名" style={{ flex: 1, height: 26, fontSize: 13 }} />
                      <button onClick={() => removeScenario(i)} disabled={optScenarios.length <= 1} style={{ ...btn, height: 26, background: "transparent", color: optScenarios.length <= 1 ? "#475569" : "#f87171", border: "none", cursor: optScenarios.length <= 1 ? "not-allowed" : "pointer" }}>✕</button>
                    </div>
                    <textarea value={s.directive} onChange={e => updateScenario(i, "directive", e.target.value)} placeholder="该方案的战略指令（如：坚决反对招安，独立发展）" style={{ height: 44, fontSize: 13, width: "100%" }} />
                    <input list={`ents-${i}`} value={s.entity_ref} onChange={e => updateScenario(i, "entity_ref", e.target.value)} placeholder="我方实体（留空=全体存活率；量化模式按此判胜）" style={{ height: 26, fontSize: 13, width: "100%", marginTop: 4 }} />
                    <datalist id={`ents-${i}`}>
                      {(graphData?.nodes || []).map(n => <option key={n.id} value={n.name} />)}
                    </datalist>
                  </div>
                ))}
                <button onClick={addScenario} style={{ ...btn, width: "100%", background: "#1e293b", color: "#cbd5e1" }}>＋ 添加方案</button>
              </div>
              <div style={{ fontSize: 12, color: "#64748b" }}>
                ⏱ 共 {optScenarios.length} 方案 × {optIterations} 次 = {optScenarios.length * optIterations} 次完整推演
                <span style={{ display: "block", fontSize: 11, color: "#475569" }}>本地 LM Studio 串行排队，单次约数分钟，建议先用小次数试跑</span>
              </div>
            </div>
          )}
        </div>

        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>会话列表（历史推演记录）</div>
        {sessions.length === 0 && (
          <div style={{ color: "#94a3b8", fontSize: 13, textAlign: "center", padding: 20 }}>
            暂无推演会话，请创建新会话开始
          </div>
        )}
        {sessions.map(s => {
          const running = ["ontology_running", "graph_running", "agents_running", "simulating", "reporting"].includes(s.status);
          return (
            <div
              key={s.id}
              onClick={() => selectSession(s.id)}
              style={{
                padding: "8px 10px", marginBottom: 6, borderRadius: 8, cursor: "pointer",
                background: selectedId === s.id ? "#1e3a8a" : "#1e293b",
                border: "1px solid " + (selectedId === s.id ? "#3b82f6" : "#334155"),
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {s.title || s.id.slice(0, 8)}
                </span>
                <button
                  title={running ? "推演进行中，无法删除" : "删除记录"}
                  onClick={e => handleDelete(s.id, e)}
                  disabled={running}
                  style={{ flexShrink: 0, background: "transparent", border: "none", color: running ? "#475569" : "#f87171", cursor: running ? "not-allowed" : "pointer", fontSize: 14, lineHeight: 1, padding: "0 2px" }}
                >🗑</button>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4, alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "#cbd5e1", background: "#0f172a", borderRadius: 4, padding: "1px 6px" }}>{PHASE_LABELS[s.status] || s.status}</span>
                {s.entity_count > 0 && <span style={{ fontSize: 11, color: "#94a3b8" }}>{s.entity_count} 实体</span>}
                {s.agent_count > 0 && <span style={{ fontSize: 11, color: "#94a3b8" }}>{s.agent_count} 智能体</span>}
                {s.current_round > 0 && <span style={{ fontSize: 11, color: "#94a3b8" }}>{s.current_round}/{s.total_rounds} 轮</span>}
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{(s.created_at || "").slice(0, 19).replace("T", " ")}</div>
            </div>
          );
        })}
      </div>

      {/* ── Right Panel ── */}
      <div style={{ display: "grid", gridTemplateRows: "auto auto 1fr auto", overflow: "hidden" }}>
        {selected ? (
          <>
            <div className="topbar" style={{ minHeight: 36, padding: "4px 12px" }}>
              <div className="topbarStatusRow">
                <span className="topbarWs">{selected.title || selected.id.slice(0, 8)}</span>
                <span className="pill">{PHASE_LABELS[selected.status] || selected.status}</span>
                {selected.entity_count > 0 && <span className="pill">{selected.entity_count} 实体</span>}
                {selected.relation_count > 0 && <span className="pill">{selected.relation_count} 关系</span>}
                {selected.agent_count > 0 && <span className="pill">{selected.agent_count} 智能体</span>}
                {selected.current_round > 0 && <span className="pill">{selected.current_round}/{selected.total_rounds} 轮</span>}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <label title="启用资源分配（多动作）：每方每轮可同时把资源分配给多个动作（如进攻+防守），运行前生效" style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: "#cbd5e1", cursor: "pointer" }}>
                  <Toggle checked={optMultiAction} onChange={setOptMultiAction} />
                  多动作
                </label>
                {optMultiAction && (
                  <select value={optMaxActions} onChange={e => setOptMaxActions(parseInt(e.target.value))} title="每方最多动作数" style={{ height: 24, fontSize: 12, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 4 }}>
                    <option value={2}>最多2</option>
                    <option value={3}>最多3</option>
                    <option value={4}>最多4</option>
                  </select>
                )}
                {optEnabled ? (
                  optRunning ? (
                    <button className="btnSmall" style={{ marginRight: 6, background: "#ef4444", color: "#fff", border: "none" }} onClick={cancelOptimize}>
                      取消优化
                    </button>
                  ) : (
                    <button className="btnSmall btnSmallPrimary" style={{ marginRight: 6 }} onClick={startOptimize}
                      disabled={selected ? RUNNING_SET.has(selected.status) : false}>
                      启动优化
                    </button>
                  )
                ) : selected && RUNNING_SET.has(selected.status) ? (
                  <>
                    <span style={{ fontSize: 12, color: "#22c55e", background: "#052e16", borderRadius: 4, padding: "2px 8px", marginRight: 4 }}>
                      推演中
                    </span>
                    <button className="btnSmall" style={{ marginRight: 6, background: "#ef4444", color: "#fff", border: "none" }} onClick={handleCancel} disabled={loading}>
                      停止推演
                    </button>
                  </>
                ) : selected?.status === "paused" ? (
                  <button
                    className="btnSmall"
                    style={{ marginRight: 6, background: "#16a34a", color: "#fff", border: "none" }}
                    onClick={handleStart}
                    disabled={loading}
                  >
                    继续推演
                  </button>
                ) : selected?.status === "complete" ? (
                  <button
                    className="btnSmall btnSmallPrimary"
                    style={{ marginRight: 6 }}
                    onClick={handleStart}
                    disabled={loading}
                  >
                    重新推演
                  </button>
                ) : (
                  <button
                    className="btnSmall btnSmallPrimary"
                    style={{ marginRight: 6 }}
                    onClick={handleStart}
                    disabled={loading}
                  >
                    启动推演
                  </button>
                )}
              </div>
            </div>

            {/* 主区标签切换: 图谱 / 报告 / 日志 */}
            <div style={{ display: "flex", gap: 4, padding: "6px 12px 0" }}>
              {(["graph", "report", "logs", "dashboard", "timeline", "optimize", "token"] as const).map(k => (
                <button
                  key={k}
                  onClick={() => setMainTab(k)}
                  style={{
                    padding: "4px 16px", borderRadius: 6, fontSize: 13, cursor: "pointer",
                    border: "1px solid #334155",
                    background: mainTab === k ? "#3b82f6" : "#0f172a",
                    color: mainTab === k ? "#fff" : "#94a3b8",
                  }}
                >{k === "graph" ? "图谱" : k === "report" ? "报告" : k === "logs" ? "日志" : k === "dashboard" ? "态势" : k === "timeline" ? "时间线" : k === "token" ? "Token" : "优化"}</button>
              ))}
            </div>

            <div style={{ overflow: "auto", position: "relative", background: mainTab === "graph" ? "#0d1117" : "transparent" }}>
              {mainTab === "graph" && (
                graphData && graphData.nodes.length > 0 ? (
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}>
                    <ForceGraph3D
                      ref={graphRef}
                      graphData={{
                        nodes: graphData.nodes.map(n => ({ id: n.id, name: n.name, group: n.type, desc: n.description })),
                        links: graphData.links.map(l => ({ source: l.source, target: l.target, value: l.relation })),
                      }}
                      nodeLabel={(n: any) => `${n.name}\n${n.group}`}
                      nodeColor={(n: any) => {
                        const palette = ["#60a5fa", "#f59e0b", "#34d399", "#a78bfa", "#f87171"];
                        const hash = [...(n.group || "")].reduce((h, c) => h + c.charCodeAt(0), 0);
                        return palette[hash % palette.length];
                      }}
                      nodeVal={(n: any) => (graphData.links.filter(l => l.source === n.id || l.target === n.id).length || 1) * 2}
                      linkLabel={(l: any) => String(l.value)}
                      linkWidth={0.5}
                      backgroundColor="#0d1117"
                    />
                    <div style={{ position: "absolute", top: 8, right: 8, display: "flex", flexDirection: "column", gap: 4, zIndex: 10 }}>
                      {[
                        { label: "＋", title: "放大", onClick: () => zoomGraph(graphRef, 0.7) },
                        { label: "−", title: "缩小", onClick: () => zoomGraph(graphRef, 1.4) },
                        { label: "⊡", title: "重置视图（显示全部节点与连线）", onClick: () => resetGraph(graphRef) },
                      ].map(b => (
                        <button key={b.label} title={b.title} onClick={b.onClick}
                          style={{ width: 28, height: 28, borderRadius: 4, cursor: "pointer", background: "rgba(15,23,42,0.7)", color: "#e2e8f0", border: "1px solid #334155", fontSize: 14, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                          {b.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ color: "#64748b", textAlign: "center", paddingTop: 200, fontSize: 14 }}>
                    {selected.status === "created" ? "上传文档或粘贴原文后启动推演" : "推演进行中或暂无图谱数据..."}
                  </div>
                )
              )}

              {mainTab === "report" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 14, overflowY: "auto" }}>
                  {report ? (
                    <>
                      {/* ── 推演叙事报告 ── */}
                      {report.summary && (
                        <div style={{ marginBottom: 24, background: "#0f172a", borderRadius: 8, padding: 16, borderLeft: "3px solid #3b82f6" }}>
                          <div style={{ fontSize: 15, fontWeight: 700, color: "#60a5fa", marginBottom: 12 }}>
                            {report.quantified ? "📋 推演报告" : "📖 故事叙事"}
                          </div>
                          <div style={{ lineHeight: 2, whiteSpace: "pre-wrap", color: "#e2e8f0" }}>
                            {report.summary}
                          </div>
                        </div>
                      )}

                      {/* ── 风险预警 ── */}
                      {report.risk_alerts && report.risk_alerts.length > 0 && (
                        <details open style={{ marginBottom: 18 }}>
                          <summary style={{ fontSize: 14, fontWeight: 700, color: "#f87171", cursor: "pointer", borderLeft: "3px solid #ef4444", paddingLeft: 8, marginBottom: 6 }}>
                            ⚠️ 风险预警（{report.risk_alerts.length} 条）
                          </summary>
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {report.risk_alerts.map((r: string, i: number) => (
                              <div key={i} style={{ background: "#0f172a", borderRadius: 6, padding: "8px 12px", borderLeft: "2px solid #ef4444", fontSize: 13, lineHeight: 1.7 }}>🔴 {r}</div>
                            ))}
                          </div>
                        </details>
                      )}

                      {/* ── 策略建议 ── */}
                      {report.recommendations && report.recommendations.length > 0 && (
                        <details open style={{ marginBottom: 18 }}>
                          <summary style={{ fontSize: 14, fontWeight: 700, color: "#22c55e", cursor: "pointer", borderLeft: "3px solid #22c55e", paddingLeft: 8, marginBottom: 6 }}>
                            💡 策略建议（{report.recommendations.length} 条）
                          </summary>
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {report.recommendations.map((r: string, i: number) => (
                              <div key={i} style={{ background: "#0f172a", borderRadius: 6, padding: "8px 12px", borderLeft: "2px solid #22c55e", fontSize: 13, lineHeight: 1.7 }}>{r}</div>
                            ))}
                          </div>
                        </details>
                      )}

                      {/* ── 关键因果链 ── */}
                      {report.causal_summary && report.causal_summary.length > 0 && (
                        <details open style={{ marginBottom: 18 }}>
                          <summary style={{ fontSize: 14, fontWeight: 700, color: "#f59e0b", cursor: "pointer", borderLeft: "3px solid #f59e0b", paddingLeft: 8, marginBottom: 6 }}>
                            关键因果链（{report.causal_summary.length} 条）
                          </summary>
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {report.causal_summary.map((c: string, i: number) => (
                              <div key={i} style={{ background: "#0f172a", borderRadius: 6, padding: "8px 12px", borderLeft: "2px solid #f59e0b", fontSize: 12, lineHeight: 1.7 }}>
                                {c}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      {/* ── 三、时序因果叙事 / 角色弧光 ── */}
                      {report.stage_narratives && report.stage_narratives.length > 0 && (
                        typeof report.stage_narratives[0] === "string" ? (
                          <div style={{ marginBottom: 18, background: "#0f172a", borderRadius: 8, padding: 14, borderLeft: "3px solid #a78bfa" }}>
                            <div style={{ fontSize: 14, fontWeight: 700, color: "#c4b5fd", marginBottom: 10 }}>🎭 角色弧光</div>
                            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                              {(report.stage_narratives as unknown as string[]).map((arc: string, i: number) => (
                                <div key={i} style={{ fontSize: 13, lineHeight: 1.7, color: "#e2e8f0" }}>
                                  {arc}
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <details open style={{ marginBottom: 18 }}>
                            <summary style={{ fontSize: 14, fontWeight: 700, color: "#a78bfa", cursor: "pointer", borderLeft: "3px solid #a78bfa", paddingLeft: 8, marginBottom: 6 }}>
                              时序因果叙事（{report.stage_narratives.length} 阶段）
                            </summary>
                            {(report.stage_narratives as any[]).map((s: any, i: number) => (
                              <div key={i} style={{ marginBottom: 12, background: "#0f172a", borderRadius: 6, padding: 10, borderLeft: "2px solid #a78bfa" }}>
                                <div style={{ fontWeight: 600, color: "#c4b5fd", marginBottom: 6 }}>
                                  <span style={{ background: "#312e81", padding: "1px 8px", borderRadius: 4, fontSize: 11, marginRight: 8 }}>
                                    {s.round_range || `第${(i*3+1)}-${(i+1)*3}轮`}
                                  </span>
                                  {s.stage || `阶段${i+1}`}
                                </div>
                                {s.start_state && <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4, lineHeight: 1.6 }}>◉ 起始状态：{s.start_state}</div>}
                                {s.key_decisions && <div style={{ fontSize: 12, color: "#e2e8f0", marginBottom: 4, lineHeight: 1.6 }}>◉ 核心决策：{s.key_decisions}</div>}
                                {s.causal_logic && <div style={{ fontSize: 12, color: "#cbd5e1", marginBottom: 4, lineHeight: 1.6 }}>◉ 因果逻辑：{s.causal_logic}</div>}
                                {s.end_state && <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.6 }}>◉ 阶段终点：{s.end_state}</div>}
                              </div>
                            ))}
                          </details>
                        )
                      )}

                      {/* ── 四、偏离分析 ── */}
                      {report.deviation_analysis && report.deviation_analysis.length > 0 && (
                        <details open style={{ marginBottom: 18 }}>
                          <summary style={{ fontSize: 14, fontWeight: 700, color: "#fb923c", cursor: "pointer", borderLeft: "3px solid #fb923c", paddingLeft: 8, marginBottom: 6 }}>
                            决策偏离分析（{report.deviation_analysis.length} 次偏离）
                          </summary>
                          {report.deviation_analysis.map((d: any, i: number) => (
                            <div key={i} style={{ marginBottom: 8, background: "#0f172a", borderRadius: 6, padding: 10, borderLeft: "2px solid #fb923c" }}>
                              <div style={{ fontWeight: 600, color: "#fdba74", marginBottom: 4 }}>
                                第{d.round}轮 · {d.agent}
                                <span style={{ fontSize: 11, marginLeft: 8, background: d.deviation_level === "显著" ? "#7c2d12" : "#422006", padding: "1px 6px", borderRadius: 4, color: d.deviation_level === "显著" ? "#fdba74" : "#f59e0b" }}>
                                  {d.deviation_level}偏离
                                </span>
                              </div>
                              <div style={{ fontSize: 12, color: "#e2e8f0", marginBottom: 2 }}>决策：{d.decision}</div>
                              <div style={{ fontSize: 12, color: "#94a3b8" }}>原因：{d.reason}</div>
                            </div>
                          ))}
                        </details>
                      )}

                      {/* ── 五、结论 ── */}
                      {report.conclusion && (
                        report.quantified ? (
                          <details open style={{ marginBottom: 18 }}>
                            <summary style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", cursor: "pointer", borderLeft: "3px solid #60a5fa", paddingLeft: 8, marginBottom: 6 }}>
                              结论
                            </summary>
                            <div style={{ marginBottom: 12, lineHeight: 1.8, whiteSpace: "pre-wrap" }}>
                              {report.conclusion}
                            </div>
                          </details>
                        ) : (
                          <div style={{ marginBottom: 18, background: "#0f172a", borderRadius: 8, padding: 14, borderLeft: "3px solid #60a5fa" }}>
                            <div style={{ fontSize: 14, fontWeight: 700, color: "#60a5fa", marginBottom: 10 }}>📌 故事收束</div>
                            <div style={{ lineHeight: 1.8, whiteSpace: "pre-wrap", color: "#e2e8f0", fontSize: 13 }}>
                              {report.conclusion}
                            </div>
                          </div>
                        )
                      )}

                      {/* ── 附录：关键事件（仅量化模式）── */}
                      {report.quantified && report.key_events && report.key_events.length > 0 && (
                        <details style={{ marginBottom: 18 }}>
                          <summary style={{ fontSize: 14, fontWeight: 700, color: "#64748b", cursor: "pointer", borderLeft: "3px solid #475569", paddingLeft: 8, marginBottom: 6 }}>
                            附录 · 关键事件 ({report.key_events.length} 条)
                          </summary>
                          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                            {report.key_events.map((ev, i) => {
                              const text = typeof ev === "string" ? ev : (ev?.description || JSON.stringify(ev));
                              const round = ev && typeof ev === "object" && ev.round ? `[第${ev.round}轮] ` : "";
                              const sig = ev && typeof ev === "object" && ev.significance ? `（${ev.significance}）` : "";
                              return <li key={i}>{round}{text}{sig}</li>;
                            })}
                          </ul>
                        </details>
                      )}
                    </>
                  ) : (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      {selected.status === "complete" ? "暂无报告数据" : "推演完成后将生成报告"}
                    </div>
                  )}
                </div>
              )}

               {mainTab === "dashboard" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 13, overflowY: "auto" }}>
                  {snapshot ? (
                    <>
                      {/* ── 自然语言态势简报 ── */}
                      <div style={{ marginBottom: 16, background: "#0f172a", borderRadius: 8, padding: 14, borderLeft: "3px solid #3b82f6", lineHeight: 1.7 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "#60a5fa", marginBottom: 10 }}>
                          📊 第 {snapshot.round || "?"} 轮态势简报
                          {snapshot.entity_count != null && (
                            <span style={{ fontSize: 12, fontWeight: 400, color: "#64748b", marginLeft: 12 }}>
                              共 {snapshot.entity_count} 个实体存活
                            </span>
                          )}
                        </div>
                        {(() => {
                          const groups = snapshot.groups as Record<string, { count: number; metrics: Record<string, number> }> | undefined;
                          const entities = snapshot.entities as Array<{ name: string; metrics: Record<string, number>; alive: boolean }> | undefined;
                          const alerts = snapshot.alerts as Array<{ entity: string; metric: string; value: number; threshold: number; severity: string }> | undefined;
                          const recent = snapshot.recent as Array<{ agent: string; action: string; content: string; round: number }> | undefined;
                          return (
                            <>
                              {/* 总体概况 */}
                              {groups && Object.keys(groups).length > 0 && (() => {
                                const domainNames = Object.keys(groups);
                                if (domainNames.length === 1 && domainNames[0] === "generic") return null;
                                const sortedMetrics = Object.keys(Object.values(groups)[0].metrics || {});
                                const topMetrics = sortedMetrics.slice(0, 4);
                                return (
                                  <div style={{ marginBottom: 10, fontSize: 13, color: "#cbd5e1" }}>
                                    阵营概况：{domainNames.map((d, i) => (
                                      <span key={d}>{i > 0 && "、"}<span style={{ color: ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"][i % 4] }}>{d}</span>（{groups[d].count}个实体）</span>
                                    ))}。
                                    {topMetrics.map(m => {
                                      const ranked = domainNames.map(d => ({ d, v: groups[d].metrics?.[m] || 0 })).sort((a, b) => b.v - a.v);
                                      if (ranked.length < 2 || ranked[0].v === ranked[1].v) return null;
                                      return <span key={m}>{fmtMetric(m)}：<span style={{ color: "#60a5fa" }}>{ranked[0].d}</span>（{ranked[0].v}分）领先，<span style={{ color: "#94a3b8" }}>{ranked[ranked.length - 1].d}</span>（{ranked[ranked.length - 1].v}分）落后。{" "}</span>;
                                    })}
                                  </div>
                                );
                              })()}

                              {/* 风险预警 */}
                              {alerts && alerts.length > 0 && (
                                <div style={{ marginBottom: 8 }}>
                                  <div style={{ fontSize: 13, fontWeight: 600, color: "#f59e0b", marginBottom: 4 }}>⚠️ 风险预警</div>
                                  {alerts.map((a, i) => {
                                    const gap = a.value - a.threshold;
                                    const pct = Math.round((a.value / Math.max(a.threshold, 1)) * 100);
                                    const desc = a.severity === "critical"
                                      ? `仅剩 ${a.value} 分，已跌破淘汰线（${a.threshold}），面临出局风险`
                                      : `降至 ${a.value} 分（安全线 ${a.threshold}），若不改善将进一步恶化`;
                                    return (
                                      <div key={i} style={{ color: a.severity === "critical" ? "#f87171" : "#f59e0b", fontSize: 12, marginBottom: 3 }}>
                                        {a.severity === "critical" ? "🔴" : "🟡"} {a.entity} {fmtMetric(a.metric)} — {desc}
                                      </div>
                                    );
                                  })}
                                </div>
                              )}

                              {/* 近期动态 */}
                              {recent && recent.length > 0 && (
                                <div>
                                  <div style={{ fontSize: 13, fontWeight: 600, color: "#a78bfa", marginBottom: 4 }}>📋 近期动态</div>
                                  {recent.map((r, i) => (
                                    <div key={i} style={{ fontSize: 12, color: "#94a3b8", marginBottom: 2, paddingLeft: 8, borderLeft: "2px solid #334155" }}>
                                      <span style={{ color: "#cbd5e1" }}>{r.agent}</span>
                                      <span style={{ color: "#64748b" }}> → {r.action}{r.content ? `（"${r.content.slice(0, 60)}${r.content.length > 60 ? "..." : ""}"）` : ""}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>

                      {/* ── 实体指标一览表 ── */}
                      {snapshot.entities && snapshot.entities.length > 0 && (() => {
                        const entities = snapshot.entities as Array<{ name: string; metrics: Record<string, number>; alive: boolean }>;
                        const allKeys = new Set<string>();
                        entities.forEach(e => Object.keys(e.metrics || {}).forEach(k => allKeys.add(k)));
                        const keys = Array.from(allKeys).slice(0, 6);
                        const thresholds = (snapshot as any)._thresholds as Record<string, number> | undefined;
                        const getStatus = (e: typeof entities[0]) => {
                          if (!thresholds || !e.alive) return { dot: "#6b7280", text: "已淘汰" };
                           const maxRatio = Math.max(...keys.map(k => ((e.metrics || {})[k] || 0) / Math.max(thresholds[k] || 1, 1)));
                          if (keys.some(k => ((e.metrics || {})[k] || 0) <= (thresholds[k] || 0) * 1.5))
                            return { dot: "#ef4444", text: "危急" };
                          if (keys.some(k => ((e.metrics || {})[k] || 0) <= (thresholds[k] || 0) * 2))
                            return { dot: "#f97316", text: "受创" };
                          if (keys.some(k => ((e.metrics || {})[k] || 0) <= (thresholds[k] || 0) * 3))
                            return { dot: "#f59e0b", text: "承压" };
                          if (maxRatio >= 4) return { dot: "#16a34a", text: "强势" };
                          return { dot: "#22c55e", text: "稳定" };
                        };
                        const weightedScore = (e: typeof entities[0]) => {
                          if (!thresholds || !e.metrics) return 0;
                          let num = 0, den = 0;
                          for (const [k, v] of Object.entries(e.metrics)) {
                            const th = thresholds[k] || 100;
                            const w = 1 / Math.max(th, 1);
                            num += (v / 100) * w;
                            den += w;
                          }
                          return den > 0 ? (num / den) * 100 : 0;
                        };
                        return (
                          <div style={{ marginBottom: 16 }}>
                            <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 10, borderLeft: "3px solid #22c55e", paddingLeft: 8 }}>实体态势一览</div>
                            <div style={{ background: "#0f172a", borderRadius: 8, overflow: "hidden" }}>
                              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                                <thead>
                                  <tr style={{ borderBottom: "1px solid #1e293b" }}>
                                    <th style={{ padding: "8px 10px", textAlign: "left", color: "#64748b", fontWeight: 600 }}>实体</th>
                                    {keys.map(k => <th key={k} style={{ padding: "8px 6px", textAlign: "center", color: "#64748b", fontWeight: 500, fontSize: 11 }}>{fmtMetric(k)}</th>)}
                                    <th style={{ padding: "8px 10px", textAlign: "center", color: "#64748b", fontWeight: 600, fontSize: 11 }}>状态</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {entities.map(e => {
                                    const st = getStatus(e);
                                    return (
                                      <tr key={e.name} style={{ borderBottom: "1px solid #1e293b" }}>
                                        <td style={{ padding: "7px 10px", color: e.alive ? "#e2e8f0" : "#475569", fontWeight: 500 }}>{e.name}</td>
                                        {keys.map(k => {
                                          const val = e.metrics?.[k];
                                          const th = thresholds?.[k];
                                          const warn = th != null && val != null && val <= th * 1.5;
                                          const crit = th != null && val != null && val <= th;
                                          return (
                                            <td key={k} style={{ padding: "7px 4px", textAlign: "center", color: crit ? "#f87171" : warn ? "#f59e0b" : "#94a3b8", fontWeight: crit ? 700 : 400 }}>
                                              {val != null ? val : "-"}
                                            </td>
                                          );
                                        })}
                                        <td style={{ padding: "7px 10px", textAlign: "center" }}>
                                          <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: st.dot, display: "inline-block", flexShrink: 0 }} />
                                            <span style={{ color: st.dot }}>{st.text}</span>
                                          </span>
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        );
                      })()}

                      {/* ── 趋势折线图 ── */}
                      {trendSnapshots.length >= 2 && snapshot.entities && snapshot.entities.length > 0 && (() => {
                        const thresholds = (snapshot as any)._thresholds as Record<string, number> | undefined;
                        const colors = ["#3b82f6","#ef4444","#22c55e","#f59e0b","#a78bfa","#06b6d4","#f97316","#ec4899","#84cc16","#14b8a6","#e11d48","#8b5cf6","#10b981","#f43f5e","#0ea5e9","#d946ef","#65a30d","#0891b2","#be123c","#7c3aed","#059669","#e11d48","#0284c7","#c026d3","#4d7c0f","#0e7490","#9f1239"];
                        const allEntities = (snapshot.entities as Array<{name:string,metrics:Record<string,number>,alive:boolean}>);
                        const chartH = 200; const padL = 60; const padR = 30; const padB = 30; const padT = 10;
                        const svgW = 800; const svgH = chartH + padB + padT;
                        const plotW = svgW - padL - padR;
                        const plotH = chartH;
                        const rounds = trendSnapshots.map(s => s.round);
                        const minR = Math.min(...rounds); const maxR = Math.max(...rounds);
                        const xScale = (r: number) => padL + ((r - minR) / Math.max(maxR - minR, 1)) * plotW;
                        // Compute weighted scores for all entities across all rounds
                        const entityScores: Record<string, Array<{round:number,score:number}>> = {};
                        for (const ts of trendSnapshots) {
                          for (const ent of ts.entities) {
                            if (!entityScores[ent.name]) entityScores[ent.name] = [];
                            let num = 0, den = 0;
                            for (const [k,v] of Object.entries(ent.metrics || {})) {
                              const th = thresholds?.[k] || 100;
                              const w = 1 / Math.max(th, 1);
                              num += (v / 100) * w;
                              den += w;
                            }
                            entityScores[ent.name].push({round: ts.round, score: den > 0 ? (num/den)*100 : 0});
                          }
                        }
                        const allScores = Object.values(entityScores).flatMap(a => a.map(s => s.score));
                        const minScore = Math.min(...allScores, 0);
                        const maxScore = Math.max(...allScores, 100);
                        const yScale = (s: number) => padT + plotH - ((s - minScore) / Math.max(maxScore - minScore, 1)) * plotH;
                        return (
                          <div style={{ marginBottom: 16 }}>
                            <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 10, borderLeft: "3px solid #8b5cf6", paddingLeft: 8 }}>综合得分趋势</div>
                            <div style={{ overflowX: "auto", background: "#0f172a", borderRadius: 8, padding: 8 }}>
                              <svg width={svgW} height={svgH} style={{ display: "block" }}>
                                {/* Grid */}
                                {[0,1,2,3,4].map(i => { const y = padT + (i/4)*plotH; const v = maxScore - (i/4)*(maxScore-minScore); return <line key={i} x1={padL} y1={y} x2={svgW-padR} y2={y} stroke="#1e293b" strokeWidth={1} />; })}
                                {/* Y labels */}
                                {[0,1,2,3,4].map(i => { const y = padT + (i/4)*plotH; const v = maxScore - (i/4)*(maxScore-minScore); return <text key={i} x={padL-8} y={y+4} textAnchor="end" fill="#64748b" fontSize={9}>{v.toFixed(0)}</text>; })}
                                {/* X labels */}
                                {trendSnapshots.map(s => { const x = xScale(s.round); return <text key={s.round} x={x} y={svgH-8} textAnchor="middle" fill="#64748b" fontSize={9}>R{s.round}</text>; })}
                                {/* Lines */}
                                {Object.entries(entityScores).map(([name, points], ei) => {
                                  if (points.length < 2) return null;
                                  const d = points.map((p, i) => `${i===0?'M':'L'}${xScale(p.round).toFixed(1)},${yScale(p.score).toFixed(1)}`).join(' ');
                                  const s = points[points.length-1];
                                  const color = colors[ei % colors.length];
                                  return (
                                    <g key={name}>
                                      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeOpacity={0.7} />
                                      {points.map((p,i) => <circle key={i} cx={xScale(p.round)} cy={yScale(p.score)} r={i===points.length-1?3:1.5} fill={i===points.length-1?color:"#1e293b"} stroke={color} strokeWidth={1} />)}
                                      <text x={xScale(s.round)+5} y={yScale(s.score)+4} fill={color} fontSize={9}>{name.length>6?name.slice(0,6)+"..":name}</text>
                                    </g>
                                  );
                                })}
                                {/* Legend */}
                                <text x={svgW-padR} y={padT+8} textAnchor="end" fill="#64748b" fontSize={9}>{rounds.length} rounds</text>
                              </svg>
                            </div>
                          </div>
                        );
                      })()}

                    </>
                  ) : (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      {selected?.status === "complete" ? "✅ 推演已完成，请查看「报告」Tab"
                        : selected?.status ? (["simulating", "reporting"].includes(selected.status) ? "⏳ 等待首轮推演数据..."
                        : "推演启动后将显示实时态势")
                        : "📌 请选择会话并启动推演"}
                    </div>
                  )}
                </div>
              )}

              {mainTab === "timeline" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 13, display: "flex", flexDirection: "column", height: "100%" }}>
                  <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                    {(["timeline", "causal"] as const).map(v => (
                      <button key={v} onClick={() => setTimelineView(v)} style={{ padding: "3px 12px", borderRadius: 6, fontSize: 13, cursor: "pointer", border: "1px solid #334155", background: timelineView === v ? "#3b82f6" : "#0f172a", color: timelineView === v ? "#fff" : "#94a3b8" }}>{v === "timeline" ? "时间线" : "因果图"}</button>
                    ))}
                  </div>
                  <div style={{ flex: 1, overflow: "auto" }}>
                  {timelineView === "timeline" ? (
                  timeline && (timeline.timelines.length > 0 || timeline.sequence.length > 0) ? (
                    <>
                      <div style={{ marginBottom: 18 }}>
                        <div style={{ fontSize: 13, fontWeight: 700, color: "#60a5fa", marginBottom: 6, borderLeft: "3px solid #3b82f6", paddingLeft: 8 }}>
                          智能体行动时间线
                        </div>
                        {timeline.timelines.map((t, i) => {
                          const hasFsm = t.actions.some(a => a.driver === "fsm");
                          const canOverride = selected?.status === "simulating";
                          return (
                          <div key={i} style={{ marginBottom: 10, background: "#0f172a", borderRadius: 6, padding: 8 }}>
                            <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
                              <span>{t.agent_name}</span>
                              {hasFsm && canOverride && (
                                <button onClick={() => { setOvAgent(ovAgent === t.agent_name ? null : t.agent_name); setOvAction(""); }}
                                  title="FSM 接管中，可强制指定该角色下一步动作"
                                  style={{ fontSize: 11, padding: "1px 8px", borderRadius: 6, cursor: "pointer", border: "1px solid #374151", background: "#1e293b", color: "#fbbf24" }}>⚙ 干预</button>
                              )}
                            </div>
                            {ovAgent === t.agent_name && (
                              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, margin: "6px 0", background: "#1e293b", borderRadius: 6, padding: 6 }}>
                                <input value={ovAction} onChange={e => setOvAction(e.target.value)} placeholder="强制动作(如 attack)" style={{ flex: 1, minWidth: 100, height: 26, fontSize: 12 }} />
                                <input value={ovTarget} onChange={e => setOvTarget(e.target.value)} placeholder="目标(可选)" list="ov-ents" style={{ width: 90, height: 26, fontSize: 12 }} />
                                <input type="number" min={0} max={1} step={0.1} value={ovIntensity} onChange={e => setOvIntensity(Number(e.target.value))} title="强度" style={{ width: 56, height: 26, fontSize: 12 }} />
                                <input type="number" min={1} max={20} value={ovRounds} onChange={e => setOvRounds(Math.max(1, Number(e.target.value) || 1))} title="轮数" style={{ width: 48, height: 26, fontSize: 12 }} />
                                <button onClick={() => submitFsmOverride(t.agent_name)} style={{ height: 26, fontSize: 12, padding: "0 10px", borderRadius: 6, cursor: "pointer", border: "none", background: "#3b82f6", color: "#fff" }}>强制</button>
                                <datalist id="ov-ents">{(graphData?.nodes || []).map(n => <option key={n.id} value={n.name} />)}</datalist>
                              </div>
                            )}
                            <ul style={{ margin: "4px 0 0", paddingLeft: 18, lineHeight: 1.7 }}>
                              {t.actions.map((a, j) => (
                                <li key={j}>
                                  <span style={{ color: "#a78bfa" }}>{a.action}</span>
                                  {a.driver === "fsm" ? <span style={{ color: "#64748b", fontSize: 11 }}> [FSM]</span>
                                    : a.driver === "forced" ? <span style={{ color: "#fbbf24", fontSize: 11 }}> [强制]</span> : null}
                                  {a.description ? <span> — {a.description}</span> : null}
                                  {a.effect ? <span style={{ marginLeft: 6, fontSize: 11, color: a.effect.includes("-") ? "#f87171" : "#34d399" }}>（{a.effect}）</span> : null}
                                </li>
                              ))}
                            </ul>
                          </div>
                          );
                        })}
                      </div>
                      {timeline.sequence.length > 0 && (
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6, borderLeft: "3px solid #a78bfa", paddingLeft: 8 }}>事件序列（按时间）</div>
                          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                            {timeline.sequence.map((e, i) => (
                              <li key={i}><span style={{ color: "#94a3b8" }}>{e.agent_name}</span> {e.action}{e.description ? `: ${e.description}` : ""}
                                {e.effect ? <span style={{ marginLeft: 6, fontSize: 11, color: e.effect.includes("-") ? "#f87171" : "#34d399" }}>（{e.effect}）</span> : null}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      {selected.status === "complete" ? "暂无行动时序数据" : "推演完成后将生成行动时间线"}
                    </div>
                  )
                  ) : (
                     causal && causal.nodes.length > 0 ? (
                       <>
                         <div style={{ flex: 1, minHeight: 250, marginBottom: 12, background: "#0d1117", borderRadius: 6, position: "relative" }}>
                           <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}>
                             <ForceGraph3D
                             ref={causalGraphRef}
                             graphData={{
                               nodes: causal.nodes.map(n => ({ id: n.id, name: n.label, group: n.kind })),
                               links: causal.links.map(l => ({ source: l.source, target: l.target, value: l.label })),
                             }}
                             nodeLabel={(n: any) => `${n.name}\n${n.group}`}
                             nodeColor={(n: any) => n.group === "agent" ? "#3b82f6" : n.group === "event" ? "#a78bfa" : "#f59e0b"}
                             linkLabel={(l: any) => String(l.value)}
                             linkDirectionalArrowLength={3}
                             backgroundColor="#0d1117"
                             onNodeClick={(n: any) => setSelectedCausalNode(n ? n.id : null)}
                             />
                           </div>
                           <div style={{ position: "absolute", top: 8, right: 8, display: "flex", flexDirection: "column", gap: 4, zIndex: 10 }}>
                             {[
                               { label: "＋", title: "放大", onClick: () => zoomGraph(causalGraphRef, 0.7) },
                               { label: "−", title: "缩小", onClick: () => zoomGraph(causalGraphRef, 1.4) },
                               { label: "⊡", title: "重置视图（显示全部节点与连线）", onClick: () => resetGraph(causalGraphRef) },
                             ].map(b => (
                               <button key={b.label} title={b.title} onClick={b.onClick}
                                 style={{ width: 28, height: 28, borderRadius: 4, cursor: "pointer", background: "rgba(15,23,42,0.7)", color: "#e2e8f0", border: "1px solid #334155", fontSize: 14, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                 {b.label}
                               </button>
                             ))}
                           </div>
                         </div>
                         {selectedCausalNode && (
                           <div style={{ marginBottom: 12, background: "#0f172a", borderRadius: 6, padding: 10, borderLeft: "3px solid #60a5fa" }}>
                             <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                               <span style={{ fontWeight: 600, color: "#60a5fa", fontSize: 12 }}>
                                 📌 选中：{causal.nodes.find(n => n.id === selectedCausalNode)?.label || selectedCausalNode}
                               </span>
                               <button onClick={() => setSelectedCausalNode(null)} style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: 14 }}>✕</button>
                             </div>
                              <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
                                {(() => {
                                  const node = causal.nodes.find(n => n.id === selectedCausalNode);
                                  if (!node) return null;
                                  const descParts: any[] = [];
                                  if (node.kind) descParts.push(<div key="kind">类型：{node.kind}</div>);
                                  if (node.desc) descParts.push(<div key="desc">{node.desc}</div>);
                                  // Collect connected nodes by direction
                                  const relatedLinks = causal.links.filter(l => l.source === selectedCausalNode || l.target === selectedCausalNode);
                                  if (relatedLinks.length === 0) return descParts;
                                  const incomingLinks = relatedLinks.filter(l => l.target === selectedCausalNode);
                                  const outgoingLinks = relatedLinks.filter(l => l.source === selectedCausalNode);
                                  // Helper: get top-5 connected nodes sorted by link count
                                  const getTopNodes = (links: typeof relatedLinks, pickId: (l: typeof relatedLinks[0]) => string, limit: number) => {
                                    const nodeIds = new Set(links.map(pickId));
                                    const scored = Array.from(nodeIds).map(id => {
                                      const n = causal.nodes.find(nn => nn.id === id);
                                      const count = links.filter(l => pickId(l) === id).length;
                                      return { node: n, count };
                                    }).filter(x => x.node).sort((a, b) => b.count - a.count).slice(0, limit);
                                    return scored;
                                  };
                                  // Incoming nodes (source → selected)
                                  if (incomingLinks.length > 0) {
                                    const topIn = getTopNodes(incomingLinks, l => l.source, 5);
                                    descParts.push(<div key="in-title" style={{ marginTop: 6, color: "#a78bfa", fontWeight: 600 }}>──── 来源（{topIn.length} 个节点指向此处）</div>);
                                    topIn.forEach((x, i) => {
                                      const lbl = x.node!.label + (x.node!.desc ? ` — ${x.node!.desc.slice(0, 80)}` : "");
                                      descParts.push(<div key={`in-${i}`} style={{ paddingLeft: 8 }}>• [{x.node!.kind}] {lbl}</div>);
                                    });
                                  }
                                  // Outgoing nodes (selected → target)
                                  if (outgoingLinks.length > 0) {
                                    const topOut = getTopNodes(outgoingLinks, l => l.target, 5);
                                    descParts.push(<div key="out-title" style={{ marginTop: 6, color: "#60a5fa", fontWeight: 600 }}>──── 去向（指向 {topOut.length} 个节点）</div>);
                                    topOut.forEach((x, i) => {
                                      const lbl = x.node!.label + (x.node!.desc ? ` — ${x.node!.desc.slice(0, 80)}` : "");
                                      descParts.push(<div key={`out-${i}`} style={{ paddingLeft: 8 }}>• [{x.node!.kind}] {lbl}</div>);
                                    });
                                  }
                                  return descParts;
                                })()}
                              </div>
                           </div>
                         )}
                         <div>
                           <div style={{ fontSize: 13, fontWeight: 700, color: "#f87171", marginBottom: 6, borderLeft: "3px solid #ef4444", paddingLeft: 8 }}>因果归因（源 → 目标 累计指标影响，负=致衰）</div>
                           <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                             {causal.summary.map((s, i) => (
                               <li key={i}><span style={{ color: "#94a3b8" }}>{s.source}</span> → <span style={{ color: "#94a3b8" }}>{s.target}</span>: {s.metric}{s.amount >= 0 ? "+" : ""}{s.amount}</li>
                             ))}
                           </ul>
                         </div>
                       </>
                    ) : (
                      <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                        {selected.status === "complete" ? "暂无因果数据" : "推演完成后将生成因果图"}
                      </div>
                    )
                  )}
                </div>
                </div>
              )}

              {mainTab === "logs" && (
                <div ref={logsRef} style={{ padding: 8, fontSize: 12 }}>
                  {logs.length === 0 && (
                    <div style={{ color: "#94a3b8", textAlign: "center", padding: 10 }}>暂无日志</div>
                  )}
                  {logs.map((l, i) => (
                    <div key={i} style={{ padding: "1px 0", color: "#94a3b8", fontFamily: "monospace" }}>
                      <span style={{ color: "#3b82f6", marginRight: 8 }}>[{l.phase}]</span>
                      {l.message}
                    </div>
                  ))}
                </div>
              )}

              {mainTab === "optimize" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 13 }}>
                  {optRunning && optProgress && (
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ marginBottom: 4 }}>
                        进行中：{optProgress.current}（{optProgress.done}/{optProgress.total}，当前最高胜分 {optProgress.best_win.toFixed(2)}）
                      </div>
                      <div style={{ height: 8, background: "#0f172a", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${optProgress.total ? (optProgress.done / optProgress.total * 100) : 0}%`, background: "#3b82f6" }} />
                      </div>
                    </div>
                  )}
                  {!optReport && !optRunning && (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      在左侧启用“策略优化器”，配置胜利条件与多个方案后点“启动优化”。
                    </div>
                  )}
                  {optReport && (
                    <>
                      {optReport.cancelled && (
                        <div style={{ color: "#f59e0b", marginBottom: 8 }}>
                          ⚠ 优化已取消，以下为已完成部分（{optReport.completed_runs}/{optReport.total_runs}）
                        </div>
                      )}
                      {optReport.recommended && (
                        <div style={{ marginBottom: 16, background: "#0f172a", border: "1px solid #3b82f6", borderRadius: 8, padding: 12 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#60a5fa", marginBottom: 4 }}>🏆 推荐方案：{optReport.recommended.name}</div>
                          <div style={{ fontSize: 13 }}>
                            胜率 {(optReport.recommended.win_mean * 100).toFixed(0)}%
                            （95%CI {(optReport.recommended.win_ci95[0] * 100).toFixed(0)}–{(optReport.recommended.win_ci95[1] * 100).toFixed(0)}%）
                            · 成功率 {(optReport.recommended.success_rate * 100).toFixed(0)}%
                             · 成本 {optReport.recommended.cost_mean.toFixed(2)}
                           </div>
                           {optReport.recommendation_rationale && (
                             <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 6, lineHeight: 1.7 }}>
                               {optReport.recommendation_rationale}
                             </div>
                           )}
                         </div>
                      )}
                      {optReport.scenarios && optReport.scenarios.length > 0 && (
                        <div style={{ marginBottom: 16 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>胜率 / 成本 散点（帕累托前沿高亮）</div>
                          <svg width={360} height={300} style={{ background: "#0f172a", borderRadius: 8 }}>
                            <line x1={40} y1={260} x2={340} y2={260} stroke="#334155" />
                            <line x1={40} y1={20} x2={40} y2={260} stroke="#334155" />
                            {optReport.scenarios.map((s: any, i: number) => {
                              const cx = 40 + s.cost_mean * 300;
                              const cy = 260 - s.win_mean * 240;
                              return (
                                <g key={i}>
                                  <circle cx={cx} cy={cy} r={6} fill={s.is_pareto ? "#34d399" : "#64748b"} stroke="#0f172a" />
                                  <text x={cx + 8} y={cy + 4} fill="#cbd5e1" fontSize={10}>{s.name}</text>
                                </g>
                              );
                            })}
                            <text x={190} y={285} textAnchor="middle" fill="#94a3b8" fontSize={12}>成本 (Cost) →</text>
                            <text x={14} y={140} textAnchor="middle" fill="#94a3b8" fontSize={12} transform="rotate(-90, 14, 140)">胜率 (Win Rate) →</text>
                          </svg>
                        </div>
                      )}
                      <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>各方案统计</div>
                      {optReport.scenarios && optReport.scenarios.map((s: any, i: number) => (
                        <div key={i} style={{ marginBottom: 8, background: "#0f172a", borderRadius: 6, padding: 8, borderLeft: `3px solid ${s.is_pareto ? "#34d399" : "#475569"}` }}>
                          <div style={{ fontWeight: 600 }}>{s.name} {s.is_pareto && <span style={{ fontSize: 11, color: "#34d399" }}>（帕累托）</span>}</div>
                          <div style={{ fontSize: 13, color: "#94a3b8" }}>
                            胜率 {(s.win_mean * 100).toFixed(0)}% ± {((s.win_ci95 ? (s.win_ci95[1] - s.win_mean) : 0) * 100).toFixed(0)}%
                            · 成功率 {(s.success_rate * 100).toFixed(0)}%
                            · 成本 {s.cost_mean.toFixed(2)}
                            · {s.runs} 次
                          </div>
                          {s.directive && <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>指令：{s.directive}</div>}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}

              {mainTab === "token" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 13, overflowY: "auto" }}>
                  {tokenData ? (
                    <>
                      {/* 总览卡片 */}
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
                        {(() => {
                          const fmt = (n: number) => n < 10000 ? `${n}` : n < 1_000_000 ? `${(n / 1000).toFixed(1)}K` : `${(n / 1_000_000).toFixed(2)}M`;
                          const t = (n: number) => `${fmt(n)} (入${fmt(tokenData.total_prompt_tokens)}/出${fmt(tokenData.total_completion_tokens)})`;
                          return [
                          { label: "总 Tokens", value: fmt(tokenData.total_tokens), color: "#3b82f6" },
                          { label: "输入", value: fmt(tokenData.total_prompt_tokens), color: "#8b5cf6" },
                          { label: "输出", value: fmt(tokenData.total_completion_tokens), color: "#06b6d4" },
                          { label: "输入/输出比", value: tokenData.total_prompt_tokens > 0 ? `1:${(tokenData.total_completion_tokens / tokenData.total_prompt_tokens).toFixed(2)}` : "N/A", color: "#f59e0b" },
                        ].map(c => (
                          <div key={c.label} style={{ background: "#0f172a", borderRadius: 8, padding: 12, borderLeft: `3px solid ${c.color}` }}>
                            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>{c.label}</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: c.color }}>{c.value}</div>
                          </div>
                        ))})()}
                      </div>

                      {/* 各阶段分布 */}
                      {tokenData.phases && Object.keys(tokenData.phases).length > 0 && (
                        <div style={{ marginBottom: 20 }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 10, borderLeft: "3px solid #3b82f6", paddingLeft: 8 }}>各阶段分布</div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            {Object.entries(tokenData.phases).map(([phase, pdata]) => (
                              <div key={phase} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span style={{ width: 80, fontSize: 12, color: "#94a3b8", textAlign: "right", flexShrink: 0 }}>{phase === "ontology" ? "本体生成" : phase === "quantify" ? "量化识别" : phase === "graph" ? "图谱构建" : phase === "agents" ? "智能体工厂" : phase === "simulation" ? "模拟推演" : phase === "report" ? "报告生成" : phase === "resume" ? "续推恢复" : phase}</span>
                                <div style={{ flex: 1, height: 20, background: "#0f172a", borderRadius: 4, overflow: "hidden", display: "flex" }}>
                                   {tokenData.total_tokens > 0 && (
                                    <>
                                      <div style={{ height: "100%", background: "#8b5cf6", width: `${(pdata.prompt / tokenData.total_tokens * 100).toFixed(1)}%` }} title={`输入 ${pdata.prompt.toLocaleString()}`} />
                                      <div style={{ height: "100%", background: "#06b6d4", width: `${(pdata.completion / tokenData.total_tokens * 100).toFixed(1)}%` }} title={`输出 ${pdata.completion.toLocaleString()}`} />
                                    </>
                                  )}
                                </div>
                                <span style={{ width: 60, fontSize: 11, color: "#64748b", textAlign: "left", flexShrink: 0 }}>{pdata.total < 10000 ? pdata.total : pdata.total < 1_000_000 ? (pdata.total/1000).toFixed(1) + "K" : (pdata.total/1_000_000).toFixed(2) + "M"}</span>
                              </div>
                            ))}
                          </div>
                          <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: "#64748b" }}>
                            <span><span style={{ color: "#8b5cf6" }}>■</span> 输入</span>
                            <span><span style={{ color: "#06b6d4" }}>■</span> 输出</span>
                          </div>
                        </div>
                      )}

                      {/* 轮次柱状图 */}
                      {tokenData.rounds && Object.keys(tokenData.rounds).length > 0 && (
                        <div style={{ marginBottom: 20 }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 10, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>Token 消耗趋势 (每轮)</div>
                           {(() => {
                             const rounds = Object.entries(tokenData.rounds);
                             if (rounds.length === 0) return null;
                             const maxTotal = Math.max(...rounds.map(([, r]) => r.total), 1);
                             const barW = Math.max(12, Math.min(40, 800 / rounds.length));
                             const svgH = 160; const svgW = rounds.length * (barW + 6) + 40;
                             const chartH = 120; const padL = 50; const padB = 30;
                             // Dynamic label skipping to avoid overlap
                             const labelEvery = rounds.length <= 10 ? 1
                               : rounds.length <= 20 ? 2
                               : rounds.length <= 50 ? 5
                               : 10;
                            return (
                              <div style={{ overflowX: "auto" }}>
                                <svg width={svgW} height={svgH} style={{ display: "block" }}>
                                  {/* Y axis labels */}
                                  {[0, 1, 2, 3, 4].map(i => {
                                    const y = chartH - (i / 4) * chartH + 10;
                                    const val = maxTotal < 10000 ? (maxTotal * i / 4).toFixed(0)
                                      : maxTotal < 1_000_000 ? ((maxTotal / 1000) * i / 4).toFixed(1) + "K"
                                      : ((maxTotal / 1_000_000) * i / 4).toFixed(1) + "M";
                                    return <text key={i} x={padL - 8} y={y + 4} textAnchor="end" fill="#64748b" fontSize={10}>{val}</text>;
                                  })}
                                  {/* Grid lines */}
                                  {[0, 1, 2, 3, 4].map(i => {
                                    const y = chartH - (i / 4) * chartH + 10;
                                    return <line key={i} x1={padL} y1={y} x2={svgW} y2={y} stroke="#1e293b" strokeWidth={1} />;
                                  })}
                                  {rounds.map(([rnd, rdata], idx) => {
                                    const x = padL + idx * (barW + 6);
                                    const hPrompt = maxTotal > 0 ? (rdata.prompt / maxTotal) * chartH : 0;
                                    const hCompl = maxTotal > 0 ? (rdata.completion / maxTotal) * chartH : 0;
                                    const yBase = chartH + 10;
                                    return (
                                      <g key={rnd}>
                                        <rect x={x} y={yBase - hPrompt - hCompl} width={barW} height={hPrompt + hCompl} fill="#1e293b" rx={2} />
                                        <rect x={x} y={yBase - hPrompt - hCompl} width={barW} height={hPrompt} fill="#8b5cf6" rx={2} />
                                        <rect x={x} y={yBase - hCompl} width={barW} height={hCompl} fill="#06b6d4" rx={2} />
                                        {idx % labelEvery === 0 && (
                                          <text x={x + barW / 2} y={yBase + 14} textAnchor="middle" fill="#64748b" fontSize={9}>R{rnd}</text>
                                        )}
                                        <title>{`R${rnd}: 入${rdata.prompt.toLocaleString()} 出${rdata.completion.toLocaleString()} 合计${rdata.total.toLocaleString()}`}</title>
                                      </g>
                                    );
                                  })}
                                </svg>
                              </div>
                            );
                          })()}
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      {selected?.status && RUNNING_SET.has(selected.status) ? "等待 LLM 调用统计..." : "暂无 Token 统计数据"}
                    </div>
                  )}
                </div>
              )}
            </div>

            {selected.status === "simulating" && (
              <div style={{ display: "flex", gap: 6, padding: "6px 12px", borderTop: "1px solid #374151", background: "#1e293b" }}>
                <input
                  style={{ flex: 1, height: 28, fontSize: 13, width: "100%" }}
                  placeholder="输入干预指令（例如：全体转向保守策略）"
                  value={interventionText}
                  onChange={e => setInterventionText(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") sendIntervention(); }}
                />
                <button
                  className="btnSmall btnSmallPrimary"
                  style={{ height: 28, fontSize: 12 }}
                  onClick={sendIntervention}
                  disabled={sending || !interventionText.trim()}
                >
                  {sending ? "发送中..." : "发送干预"}
                </button>
              </div>
            )}
          </>
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#94a3b8", fontSize: 14 }}>
            请选择一个推演会话以开始
          </div>
        )}
      </div>

      {/* ── Settings Overlay ── */}
      {showSettings && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowSettings(false)}>
          <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, width: 520, maxHeight: "80vh", overflow: "auto", border: "1px solid #334155" }} onClick={e => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 18 }}>LLM / 嵌入模型配置</h2>
              <button onClick={() => setShowSettings(false)} style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: 20 }}>✕</button>
            </div>

            {/* Tabs */}
            <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
              <button onClick={() => setSettingsTab("llm")} style={{ flex: 1, padding: "6px 0", borderRadius: 6, border: "1px solid #334155", background: settingsTab === "llm" ? "#3b82f6" : "#0f172a", color: settingsTab === "llm" ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: 13 }}>LLM 对话模型</button>
              <button onClick={() => setSettingsTab("embed")} style={{ flex: 1, padding: "6px 0", borderRadius: 6, border: "1px solid #334155", background: settingsTab === "embed" ? "#3b82f6" : "#0f172a", color: settingsTab === "embed" ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: 13 }}>嵌入模型</button>
              <button onClick={() => setSettingsTab("engine")} style={{ flex: 1, padding: "6px 0", borderRadius: 6, border: "1px solid #334155", background: settingsTab === "engine" ? "#3b82f6" : "#0f172a", color: settingsTab === "engine" ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: 13 }}>引擎</button>
            </div>

            {settingsTab === "llm" && (
              <>
                <label style={lbl}>服务商</label>
                {cfgProviders.length === 0 ? (
                  <div style={{ color: "#f59e0b", fontSize: 13, marginBottom: 8 }}>⚠ 无法加载服务商列表 — 请确认后端已启动 (http://127.0.0.1:8000/health)</div>
                ) : (
                <select value={cfgLLMProvider} onChange={e => { setCfgLLMProvider(e.target.value); const p = cfgProviders.find(x => x.slug === e.target.value); if (p?.default_llm_base_url) { setCfgLLMBase(p.default_llm_base_url); setCfgLLMTest(""); } }} style={{ ...inp, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6 }}>
                  <option value="">选择服务商...</option>
                  {cfgProviders.map(p => <option key={p.slug} value={p.slug}>{p.name}{p.note ? ` (${p.note})` : ""}</option>)}
                </select>
                )}

                <label style={lbl}>API 地址</label>
                <input style={inp} value={cfgLLMBase} onChange={e => setCfgLLMBase(e.target.value)} placeholder="http://127.0.0.1:1234/v1" />
                <div style={{ marginTop: 4, marginBottom: 12, display: "flex", gap: 8 }}>
                  <button onClick={testLLM} disabled={cfgLLMTest === "testing"} style={{ ...btn, background: "#334155", color: "#e2e8f0" }}>
                    {cfgLLMTest === "testing" ? "测试中..." : cfgLLMTest === "ok" ? "✓ 连接成功" : cfgLLMTest === "fail" ? "✗ 连接失败" : "测试连接"}
                  </button>
                </div>

                <label style={lbl}>API Key</label>
                <input style={inp} type="password" value={cfgLLMKey} onChange={e => setCfgLLMKey(e.target.value)} placeholder="sk-... (LM Studio 无需填写)" />

                <label style={lbl}>模型名称</label>
                <input style={inp} value={cfgLLMModel} onChange={e => setCfgLLMModel(e.target.value)} placeholder="qwen/qwen3.5-9b" />
                <div style={{ marginTop: 4, marginBottom: 12 }}>
                  <button onClick={fetchModels} disabled={cfgFetchingModels} style={{ ...btn, background: "#334155", color: "#e2e8f0" }}>{cfgFetchingModels ? "获取中..." : "拉取模型列表"}</button>
                </div>
              </>
            )}
            {settingsTab === "embed" && (
              <>
                <label style={lbl}>服务商</label>
                <select value={cfgEmbedProvider} onChange={e => { setCfgEmbedProvider(e.target.value); const p = cfgProviders.find(x => x.slug === e.target.value); if (p?.default_llm_base_url) { setCfgEmbedBase(p.default_llm_base_url); } }} style={{ ...inp, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6 }}>
                  <option value="">独立配置...</option>
                  {cfgProviders.map(p => <option key={p.slug} value={p.slug}>{p.name}{p.default_embed_model ? ` (${p.default_embed_model})` : ""}{p.note ? ` — ${p.note}` : ""}</option>)}
                </select>

                <label style={lbl}>嵌入 API 地址</label>
                <input style={inp} value={cfgEmbedBase} onChange={e => setCfgEmbedBase(e.target.value)} placeholder="http://127.0.0.1:1234/v1" />

                <label style={lbl}>嵌入 API Key</label>
                <input style={inp} type="password" value={cfgEmbedKey} onChange={e => setCfgEmbedKey(e.target.value)} placeholder="嵌入服务 API Key" />

                <label style={lbl}>嵌入模型名称</label>
                <input style={inp} value={cfgEmbedModel} onChange={e => setCfgEmbedModel(e.target.value)} placeholder="text-embedding-3-small" />
                <div style={{ marginTop: 4, marginBottom: 12 }}>
                  <button onClick={fetchModels} disabled={cfgFetchingModels} style={{ ...btn, background: "#334155", color: "#e2e8f0" }}>{cfgFetchingModels ? "获取中..." : "拉取模型列表"}</button>
                </div>
              </>
            )}

            {/* LLM Model list */}
            {settingsTab === "llm" && cfgLLMFetchedModels.length > 0 && (
              <div style={{ maxHeight: 180, overflow: "auto", marginBottom: 16, background: "#0f172a", borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>对话模型 ({cfgLLMFetchedModels.length})</div>
                {cfgLLMFetchedModels.map(m => (
                  <div key={m} onClick={() => setCfgLLMModel(m)} style={{ padding: "3px 6px", cursor: "pointer", borderRadius: 4, fontSize: 13, color: cfgLLMModel === m ? "#3b82f6" : "#cbd5e1" }}>{m}</div>
                ))}
              </div>
            )}
            {settingsTab === "llm" && cfgLLMModelError && <div style={{ color: "#ef4444", fontSize: 13, marginBottom: 12 }}>{cfgLLMModelError}</div>}
            {/* Embed Model list */}
            {settingsTab === "embed" && cfgEmbedFetchedModels.length > 0 && (
              <div style={{ maxHeight: 180, overflow: "auto", marginBottom: 16, background: "#0f172a", borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>嵌入模型 ({cfgEmbedFetchedModels.length})</div>
                {cfgEmbedFetchedModels.map(m => (
                  <div key={m} onClick={() => setCfgEmbedModel(m)} style={{ padding: "3px 6px", cursor: "pointer", borderRadius: 4, fontSize: 13, color: cfgEmbedModel === m ? "#3b82f6" : "#cbd5e1" }}>{m}</div>
                ))}
              </div>
            )}
            {settingsTab === "embed" && cfgEmbedModelError && <div style={{ color: "#ef4444", fontSize: 13, marginBottom: 12 }}>{cfgEmbedModelError}</div>}

            {/* ── 引擎配置 Tab ── */}
            {settingsTab === "engine" && (
              <>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginBottom: 8, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>推演控制</div>
                <label style={lbl}>默认轮数 <span style={{ color: "#64748b" }}>— 创建会话时的预设推演轮数</span></label>
                <input style={inp} type="number" min={1} max={100} value={cfgDefaultRounds} onChange={e => setCfgDefaultRounds(Math.max(1, Number(e.target.value) || 1))} />
                <label style={lbl}>最大智能体 <span style={{ color: "#64748b" }}>— 单次推演智能体数量上限</span></label>
                <input style={inp} type="number" min={1} max={50000} value={cfgMaxAgents} onChange={e => setCfgMaxAgents(Math.max(1, Number(e.target.value) || 1))} />
                <label style={lbl}>候选策略数 <span style={{ color: "#64748b" }}>— LLM 每条决策生成的候选策略数</span></label>
                <input style={inp} type="number" min={1} max={10} value={cfgCandidateCount} onChange={e => setCfgCandidateCount(Math.max(1, Number(e.target.value) || 1))} />

                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginTop: 12, marginBottom: 8, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>LLM 行为</div>
                <label style={lbl}>决策温度: {cfgLLMTemp.toFixed(1)} <span style={{ color: "#64748b" }}>— 越高决策越随机</span></label>
                <input type="range" min={0} max={1} step={0.1} value={cfgLLMTemp} onChange={e => setCfgLLMTemp(parseFloat(e.target.value))} style={{ width: "100%" }} />
                <label style={lbl}>并发上限 <span style={{ color: "#64748b" }}>— 同时进行的 LLM 请求数</span></label>
                <input style={inp} type="number" min={1} max={16} value={cfgMaxConcurrent} onChange={e => setCfgMaxConcurrent(Math.max(1, Number(e.target.value) || 1))} />
                <label style={lbl}>重试次数 <span style={{ color: "#64748b" }}>— 连接失败额外重试次数</span></label>
                <input style={inp} type="number" min={0} max={10} value={cfgRetryPasses} onChange={e => setCfgRetryPasses(Math.max(0, Number(e.target.value) || 0))} />
                <label style={lbl}>故障中断比: {cfgFailThreshold.toFixed(2)} <span style={{ color: "#64748b" }}>— agent 故障超此比例则中断推演</span></label>
                <input type="range" min={0} max={1} step={0.05} value={cfgFailThreshold} onChange={e => setCfgFailThreshold(parseFloat(e.target.value))} style={{ width: "100%" }} />

                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginTop: 12, marginBottom: 8, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>检索</div>
                <label style={lbl}>Top-K <span style={{ color: "#64748b" }}>— 语义检索返回片段数</span></label>
                <input style={inp} type="number" min={1} max={30} value={cfgRetrieveTopK} onChange={e => setCfgRetrieveTopK(Math.max(1, Number(e.target.value) || 1))} />
                <label style={lbl}>相似度阈值: {cfgSimilarity.toFixed(2)} <span style={{ color: "#64748b" }}>— 检索匹配最低相似度</span></label>
                <input type="range" min={0} max={1} step={0.05} value={cfgSimilarity} onChange={e => setCfgSimilarity(parseFloat(e.target.value))} style={{ width: "100%" }} />
                <label style={{ ...lbl, display: "flex", alignItems: "center", gap: 8 }}>混合检索 <span style={{ color: "#64748b" }}>— LanceDB 向量+全文混合</span><Toggle checked={cfgEventHybrid} onChange={setCfgEventHybrid} /></label>
                <label style={{ ...lbl, display: "flex", alignItems: "center", gap: 8 }}>关系增强 <span style={{ color: "#64748b" }}>— Kuzu 邻居增强检索</span><Toggle checked={cfgRecallBoost} onChange={setCfgRecallBoost} /></label>

                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginTop: 12, marginBottom: 8, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>超时</div>
                <label style={lbl}>连接超时(秒) <span style={{ color: "#64748b" }}>— HTTP 连接握手超时</span></label>
                <input style={inp} type="number" min={5} max={300} value={cfgConnectTimeout} onChange={e => setCfgConnectTimeout(Math.max(5, Number(e.target.value) || 5))} />
                <label style={lbl}>生成超时(秒) <span style={{ color: "#64748b" }}>— 单次 LLM 生成响应超时</span></label>
                <input style={inp} type="number" min={30} max={7200} value={cfgGenTimeout} onChange={e => setCfgGenTimeout(Math.max(30, Number(e.target.value) || 30))} />

                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginTop: 12, marginBottom: 8, borderLeft: "3px solid #f59e0b", paddingLeft: 8 }}>实体</div>
                <label style={{ ...lbl, display: "flex", alignItems: "center", gap: 8 }}>安全网 <span style={{ color: "#64748b" }}>— 过滤非独立决策实体(部门/职务)</span><Toggle checked={cfgSafetyNet} onChange={setCfgSafetyNet} /></label>
              </>
            )}

            <button onClick={saveConfig} disabled={cfgSaving} style={{ ...btn, width: "100%", background: "#3b82f6", color: "#fff", height: 36, fontSize: 14 }}>
              {cfgSaving ? "保存中..." : "保存配置"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
