export interface SessionItem {
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

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  description: string;
}

export interface GraphLink {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface LogEntry {
  phase: string;
  message: string;
  timestamp: string;
}
