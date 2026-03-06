export interface ThinkingEvent {
  type: "thinking";
  content: string;
  elapsed_s: number;
  turn: number;
}

export interface ToolCallEvent {
  type: "tool_call";
  turn: number;
  tool: string;
  model_id: string;
  display_name: string;
  diagram_key: string;
  orch_latency_ms: number;
}

export interface SearchResultEvent {
  type: "search_result";
  turn: number;
  tool: string;
  specialist: string;
  query: string;
  count: number;
  latency_ms: number;
  in_tokens: number;
  out_tokens: number;
  est_cost_usd: number;
  diagram_key: string;
  display_name: string;
  total_cost: number;
}

export interface ReasoningResultEvent {
  type: "reasoning_result";
  turn: number;
  tool: string;
  specialist: string;
  latency_ms: number;
  in_tokens: number;
  out_tokens: number;
  code_executed: boolean;
  code_preview: string;
  exec_output: string;
  est_cost_usd: number;
  diagram_key: string;
  display_name: string;
  total_cost: number;
}

export interface AnswerEvent {
  type: "answer_attempt" | "answer_final";
  turn: number;
  tool: string;
  specialist: string;
  latency_ms: number;
  in_tokens: number;
  out_tokens: number;
  prediction: string;
  is_final: boolean;
  est_cost_usd: number;
  diagram_key: string;
  display_name: string;
  total_cost: number;
}

export interface DoneEvent {
  type: "done";
  final_answer: string;
  total_turns: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_cost: number;
  tool_trace: TraceEntry[];
}

export interface StatusEvent {
  type: "status";
  message: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type SSEEvent =
  | ThinkingEvent
  | ToolCallEvent
  | SearchResultEvent
  | ReasoningResultEvent
  | AnswerEvent
  | DoneEvent
  | StatusEvent
  | ErrorEvent;

export interface TraceEntry {
  turn: number;
  tool: string;
  specialist?: string;
  query?: string;
  count?: number;
  latency_ms?: number;
  in_tokens?: number;
  out_tokens?: number;
  code_executed?: boolean;
  est_cost_usd?: number;
  note?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "thinking" | "tool_call" | "search" | "reasoning" | "answer_attempt" | "answer_final" | "final" | "status" | "error";
  content: string;
  tool?: string;
  model_id?: string;
  display_name?: string;
  diagram_key?: string;
  latency_ms?: number;
  orch_latency_ms?: number;
  in_tokens?: number;
  out_tokens?: number;
  count?: number;
  query?: string;
  code_preview?: string;
  exec_output?: string;
  is_final?: boolean;
  prediction?: string;
  total_turns?: number;
  total_tokens?: number;
  total_cost?: number;
  elapsed_s?: number;
}

export interface DiagramState {
  activeModel: string | null;
  activeTool: string | null;
  modelId: string;
  turnHistory: Array<{
    turn: number;
    tool: string;
    display: string;
    latency_ms: number;
  }>;
}
