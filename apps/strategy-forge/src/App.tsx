import { useState, useEffect, useCallback, useRef } from "react";
import ForceGraph3D from "react-force-graph-3d";

const API_BASE = import.meta.env.DEV ? "/api/forge" : "http://127.0.0.1:8000/api/forge";

const lbl: React.CSSProperties = { fontSize: 12, color: "#94a3b8", marginBottom: 4, display: "block" };
const inp: React.CSSProperties = { height: 32, marginBottom: 8, width: "100%" };
const btn: React.CSSProperties = { height: 28, fontSize: 12, borderRadius: 6, border: "1px solid #334155", cursor: "pointer", padding: "0 12px" };

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

interface ReportData {
  summary?: string;
  key_events?: Array<any>;
  risk_alerts?: string[];
  recommendations?: string[];
}

// ── Phase Labels ──

const PHASE_LABELS: Record<string, string> = {
  created: "已创建",
  ontology_running: "本体生成中...",
  graph_running: "图谱构建中...",
  agents_running: "智能体生成中...",
  simulating: "模拟推演中...",
  reporting: "报告生成中...",
  complete: "已完成",
  failed: "失败",
  paused: "已暂停",
};

// ── Main App ──

export default function App() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [sourceMaterial, setSourceMaterial] = useState("");
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [report, setReport] = useState<ReportData | null>(null);
  const [mainTab, setMainTab] = useState<"graph" | "report" | "logs">("graph");
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [preGoal, setPreGoal] = useState("");
  const [interventionText, setInterventionText] = useState("");
  const [sending, setSending] = useState(false);
  const logsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Settings ──
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"llm" | "embed">("llm");
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
  const [cfgFetchedModels, setCfgFetchedModels] = useState<string[]>([]);
  const [cfgModelError, setCfgModelError] = useState("");
  const [cfgSaving, setCfgSaving] = useState(false);
  const [cfgLLMTest, setCfgLLMTest] = useState<"" | "testing" | "ok" | "fail">("");
  const [cfgProviders, setCfgProviders] = useState<Array<{slug:string;name:string;default_llm_base_url:string;default_llm_model:string;default_embed_model:string;note:string}>>([]);

  const fetchConfig = useCallback(async (): Promise<boolean> => {
    try {
      const [lr, er, pr] = await Promise.all([
        fetch(`${API_BASE}/config/llm`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/config/embedding`).then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/config/providers`).then(r => r.ok ? r.json() : null),
      ]);
      if (lr) { setCfgLLMBase(lr.llm_base_url || ""); setCfgLLMKey(lr.llm_api_key || ""); setCfgLLMModel(lr.llm_model || ""); setCfgLLMProvider(lr.provider_slug || ""); setCfgLLMTemp(lr.llm_temperature || 0.3); }
      if (er) { setCfgEmbedBase(er.embedding_api_base || ""); setCfgEmbedKey(er.embedding_api_key || ""); setCfgEmbedModel(er.embedding_model_name || ""); setCfgEmbedProvider(er.provider_slug || ""); }
      if (pr) { setCfgProviders(pr.providers || []); }
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

  const fetchModels = useCallback(async () => {
    const base = settingsTab === "llm" ? cfgLLMBase : cfgEmbedBase;
    const key = settingsTab === "llm" ? cfgLLMKey : cfgEmbedKey;
    if (!base) return;
    setCfgFetchingModels(true); setCfgModelError(""); setCfgFetchedModels([]);
    try {
      const r = await fetch(`${API_BASE}/config/list-models`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_type: "openai", base_url: base, api_key: key || "local" }),
      });
      const data = await r.json();
      if (data.error) { setCfgModelError(data.error); }
      else { setCfgFetchedModels(data.models || []); }
    } catch (e: any) { setCfgModelError(e.message || "Failed"); }
    setCfgFetchingModels(false);
  }, [settingsTab, cfgLLMBase, cfgLLMKey, cfgEmbedBase, cfgEmbedKey]);

  const testLLM = useCallback(async () => {
    setCfgLLMTest("testing");
    try {
      const r = await fetch(`${API_BASE}/config/test-connection`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_type: "openai", base_url: cfgLLMBase, api_key: cfgLLMKey || "local" }),
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
        body: JSON.stringify({ embedding_api_base: cfgEmbedBase || cfgLLMBase, embedding_api_key: cfgEmbedKey || cfgLLMKey, embedding_model_name: cfgEmbedModel, provider_slug: cfgEmbedProvider || cfgLLMProvider }),
      });
      await fetchConfig();
      setShowSettings(false);
    } catch { /* ignore */ }
    setCfgSaving(false);
  }, [cfgLLMBase, cfgLLMKey, cfgLLMModel, cfgLLMProvider, cfgLLMTemp, cfgEmbedBase, cfgEmbedKey, cfgEmbedModel, cfgEmbedProvider, fetchConfig]);

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
  }, [fetchGraph, fetchLogs, fetchReport]);

  const handleCreate = useCallback(async () => {
    if (!title.trim() || !sourceMaterial.trim()) return;
    setCreating(true);
    try {
      const r = await fetch(`${API_BASE}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, source_material: sourceMaterial }),
      });
      if (r.ok) {
        const data = await r.json();
        setSelectedId(data.id);
        setSessions(prev => [{ id: data.id, title: data.title, status: data.status, phase: "", entity_count: 0, relation_count: 0, agent_count: 0, current_round: 0, total_rounds: 10, created_at: data.created_at }, ...prev]);
      }
    } catch { /* ignore */ }
    setCreating(false);
  }, [title, sourceMaterial]);

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
      const r = await fetch(`${API_BASE}/session/${selectedId}/start`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      await fetchSessions();
      await fetchGraph(selectedId);
      await fetchLogs(selectedId);
      await fetchReport(selectedId);
      setMainTab("report");
    } catch (e: any) {
      alert("推演启动失败: " + (e.message || "未知错误"));
    }
    setLoading(false);
  }, [selectedId, fetchSessions, fetchGraph, fetchLogs, fetchReport]);

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

  // SSE auto-refresh logs during simulation
  useEffect(() => {
    if (!selectedId) return;
    const selected = sessions.find(s => s.id === selectedId);
    if (!selected || selected.status !== "simulating") return;
    const es = new EventSource(`${API_BASE}/session/${selectedId}/stream`);
    es.onmessage = (ev: MessageEvent) => {
      if (ev.data === "[DONE]") { es.close(); fetchSessions(); fetchGraph(selectedId); fetchReport(selectedId); return; }
      try {
        const d = JSON.parse(ev.data);
        setLogs(prev => [...prev.slice(-200), { phase: d.phase || d.type || "", message: d.message || "", timestamp: d.timestamp || "" }]);
        if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
      } catch { /* ignore */ }
    };
    es.onerror = () => { es.close(); };
    return () => es.close();
  }, [selectedId, sessions, fetchSessions, fetchGraph]);

  const selected = sessions.find(s => s.id === selectedId);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", height: "100%", overflow: "hidden" }}>
      {/* ── Left Panel: Sessions ── */}
      <div style={{ borderRight: "1px solid #374151", overflow: "auto", padding: 12 }}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>StrategyForge 战略推演</span>
          <button onClick={() => { fetchConfig(); setShowSettings(true); }} style={{ background: "#334155", border: "1px solid #475569", color: "#e2e8f0", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontSize: 11 }}>⚙ 配置</button>
        </h3>

        <div className="card" style={{ marginBottom: 10 }}>
          <input
            style={{ height: 32, marginBottom: 6, width: "100%" }}
            placeholder="会话标题"
            value={title}
            onChange={e => setTitle(e.target.value)}
          />
          <textarea
            style={{ height: 100, fontSize: 12, marginBottom: 6, width: "100%" }}
            placeholder="粘贴种子材料（或点击上传文档）"
            value={sourceMaterial}
            onChange={e => setSourceMaterial(e.target.value)}
          />
          <textarea
            style={{ height: 48, fontSize: 12, marginBottom: 6, width: "100%" }}
            placeholder="推演前愿景/目标（可选）"
            value={preGoal}
            onChange={e => setPreGoal(e.target.value)}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.json,.pdf,.docx,.py,.js,.ts,.rs,.go,.java,.c,.cpp,.csv,.log,.yaml,.yml"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          <button
            style={{ width: "100%", height: 28, fontSize: 12, marginBottom: 6, background: "#1e293b", border: "1px solid #374151", borderRadius: 6, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, color: "#94a3b8" }}
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

        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>会话列表（历史推演记录）</div>
        {sessions.length === 0 && (
          <div style={{ color: "#94a3b8", fontSize: 12, textAlign: "center", padding: 20 }}>
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
                <span style={{ fontSize: 10, color: "#cbd5e1", background: "#0f172a", borderRadius: 4, padding: "1px 6px" }}>{PHASE_LABELS[s.status] || s.status}</span>
                {s.entity_count > 0 && <span style={{ fontSize: 10, color: "#94a3b8" }}>{s.entity_count} 实体</span>}
                {s.agent_count > 0 && <span style={{ fontSize: 10, color: "#94a3b8" }}>{s.agent_count} 智能体</span>}
                {s.current_round > 0 && <span style={{ fontSize: 10, color: "#94a3b8" }}>{s.current_round}/{s.total_rounds} 轮</span>}
              </div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>{(s.created_at || "").slice(0, 19).replace("T", " ")}</div>
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
              <div>
                <button
                  className="btnSmall btnSmallPrimary"
                  style={{ marginRight: 6 }}
                  onClick={handleStart}
                  disabled={loading || selected.status === "simulating"}
                >
                  {selected.status === "complete" ? "重新推演" : loading ? "运行中..." : "启动推演"}
                </button>
              </div>
            </div>

            {/* 主区标签切换: 图谱 / 报告 / 日志 */}
            <div style={{ display: "flex", gap: 4, padding: "6px 12px 0" }}>
              {(["graph", "report", "logs"] as const).map(k => (
                <button
                  key={k}
                  onClick={() => setMainTab(k)}
                  style={{
                    padding: "4px 16px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                    border: "1px solid #334155",
                    background: mainTab === k ? "#3b82f6" : "#0f172a",
                    color: mainTab === k ? "#fff" : "#94a3b8",
                  }}
                >{k === "graph" ? "图谱" : k === "report" ? "报告" : "日志"}</button>
              ))}
            </div>

            <div style={{ overflow: "auto", position: "relative", background: mainTab === "graph" ? "#0d1117" : "transparent" }}>
              {mainTab === "graph" && (
                graphData && graphData.nodes.length > 0 ? (
                  <ForceGraph3D
                    graphData={{
                      nodes: graphData.nodes.map(n => ({ id: n.id, name: n.name, group: n.type, desc: n.description })),
                      links: graphData.links.map(l => ({ source: l.source, target: l.target, value: l.relation })),
                    }}
                    nodeLabel={(n: any) => `${n.name}\n${n.group}`}
                    nodeColor={(n: any) => {
                      const colors: Record<string, string> = { Person: "#60a5fa", Organization: "#f59e0b", Event: "#ef4444", Concept: "#34d399", Location: "#a78bfa" };
                      return colors[n.group] || "#94a3b8";
                    }}
                    nodeVal={(n: any) => (graphData.links.filter(l => l.source === n.id || l.target === n.id).length || 1) * 2}
                    linkLabel={(l: any) => String(l.value)}
                    linkWidth={0.5}
                    backgroundColor="#0d1117"
                    width={window.innerWidth - 340}
                    height={520}
                  />
                ) : (
                  <div style={{ color: "#64748b", textAlign: "center", paddingTop: 200, fontSize: 14 }}>
                    {selected.status === "created" ? "上传文档或粘贴原文后启动推演" : "推演进行中或暂无图谱数据..."}
                  </div>
                )
              )}

              {mainTab === "report" && (
                <div style={{ padding: 16, color: "#cbd5e1", fontSize: 13 }}>
                  {report ? (
                    <>
                      {report.summary && (
                        <div style={{ marginBottom: 18 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6, borderLeft: "3px solid #3b82f6", paddingLeft: 8 }}>推演总结</div>
                          <div style={{ lineHeight: 1.8, whiteSpace: "pre-wrap" }}>{report.summary}</div>
                        </div>
                      )}
                      {report.risk_alerts && report.risk_alerts.length > 0 && (
                        <div style={{ marginBottom: 18 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#f87171", marginBottom: 6, borderLeft: "3px solid #ef4444", paddingLeft: 8 }}>风险预警</div>
                          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                            {report.risk_alerts.map((x, i) => <li key={i}>{x}</li>)}
                          </ul>
                        </div>
                      )}
                      {report.recommendations && report.recommendations.length > 0 && (
                        <div style={{ marginBottom: 18 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#34d399", marginBottom: 6, borderLeft: "3px solid #10b981", paddingLeft: 8 }}>策略建议</div>
                          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                            {report.recommendations.map((x, i) => <li key={i}>{x}</li>)}
                          </ul>
                        </div>
                      )}
                      {report.key_events && report.key_events.length > 0 && (
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0", marginBottom: 6, borderLeft: "3px solid #a78bfa", paddingLeft: 8 }}>关键事件</div>
                          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
                            {report.key_events.map((ev, i) => {
                              const text = typeof ev === "string" ? ev : (ev?.description || JSON.stringify(ev));
                              const round = ev && typeof ev === "object" && ev.round ? `[第${ev.round}轮] ` : "";
                              const sig = ev && typeof ev === "object" && ev.significance ? `（${ev.significance}）` : "";
                              return <li key={i}>{round}{text}{sig}</li>;
                            })}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ color: "#64748b", textAlign: "center", paddingTop: 60 }}>
                      {selected.status === "complete" ? "暂无报告数据" : "推演完成后将生成报告"}
                    </div>
                  )}
                </div>
              )}

              {mainTab === "logs" && (
                <div ref={logsRef} style={{ padding: 8, fontSize: 11 }}>
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
            </div>

            {selected.status === "simulating" && (
              <div style={{ display: "flex", gap: 6, padding: "6px 12px", borderTop: "1px solid #374151", background: "#1e293b" }}>
                <input
                  style={{ flex: 1, height: 28, fontSize: 12, width: "100%" }}
                  placeholder="输入干预指令（例如：全体转向保守策略）"
                  value={interventionText}
                  onChange={e => setInterventionText(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") sendIntervention(); }}
                />
                <button
                  className="btnSmall btnSmallPrimary"
                  style={{ height: 28, fontSize: 11 }}
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
            </div>

            {settingsTab === "llm" ? (
              <>
                <label style={lbl}>服务商</label>
                {cfgProviders.length === 0 ? (
                  <div style={{ color: "#f59e0b", fontSize: 12, marginBottom: 8 }}>⚠ 无法加载服务商列表 — 请确认后端已启动 (http://127.0.0.1:8000/health)</div>
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
            ) : (
              <>
                <label style={lbl}>服务商</label>
                <select value={cfgEmbedProvider} onChange={e => { setCfgEmbedProvider(e.target.value); const p = cfgProviders.find(x => x.slug === e.target.value); if (p?.default_llm_base_url) { setCfgEmbedBase(p.default_llm_base_url); } }} style={{ ...inp, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: 6 }}>
                  <option value="">与 LLM 相同</option>
                  {cfgProviders.map(p => <option key={p.slug} value={p.slug}>{p.name}{p.default_embed_model ? ` (${p.default_embed_model})` : ""}{p.note ? ` — ${p.note}` : ""}</option>)}
                </select>

                <label style={lbl}>嵌入 API 地址</label>
                <input style={inp} value={cfgEmbedBase} onChange={e => setCfgEmbedBase(e.target.value)} placeholder={cfgLLMBase || "http://127.0.0.1:1234/v1"} />

                <label style={lbl}>嵌入 API Key</label>
                <input style={inp} type="password" value={cfgEmbedKey} onChange={e => setCfgEmbedKey(e.target.value)} placeholder="与 LLM 相同 (留空)" />

                <label style={lbl}>嵌入模型名称</label>
                <input style={inp} value={cfgEmbedModel} onChange={e => setCfgEmbedModel(e.target.value)} placeholder="text-embedding-3-small" />
                <div style={{ marginTop: 4, marginBottom: 12 }}>
                  <button onClick={fetchModels} disabled={cfgFetchingModels} style={{ ...btn, background: "#334155", color: "#e2e8f0" }}>{cfgFetchingModels ? "获取中..." : "拉取模型列表"}</button>
                </div>
              </>
            )}

            {/* Model list */}
            {cfgFetchedModels.length > 0 && (
              <div style={{ maxHeight: 180, overflow: "auto", marginBottom: 16, background: "#0f172a", borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>可用模型 ({cfgFetchedModels.length})</div>
                {cfgFetchedModels.map(m => (
                  <div key={m} onClick={() => { if (settingsTab === "llm") setCfgLLMModel(m); else setCfgEmbedModel(m); }} style={{ padding: "3px 6px", cursor: "pointer", borderRadius: 4, fontSize: 12, color: (settingsTab === "llm" ? cfgLLMModel : cfgEmbedModel) === m ? "#3b82f6" : "#cbd5e1" }}>{m}</div>
                ))}
              </div>
            )}
            {cfgModelError && <div style={{ color: "#ef4444", fontSize: 12, marginBottom: 12 }}>{cfgModelError}</div>}

            <button onClick={saveConfig} disabled={cfgSaving} style={{ ...btn, width: "100%", background: "#3b82f6", color: "#fff", height: 36, fontSize: 14 }}>
              {cfgSaving ? "保存中..." : "保存配置"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
