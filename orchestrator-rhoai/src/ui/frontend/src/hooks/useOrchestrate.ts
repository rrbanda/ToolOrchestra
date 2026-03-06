import { useCallback, useRef, useState } from "react";
import type { ChatMessage, DiagramState, SSEEvent, TraceEntry } from "../types";

let msgId = 0;
const nextId = () => `msg-${++msgId}`;

const TOOL_LABELS: Record<string, string> = {
  search: "Search",
  enhance_reasoning: "Reasoning",
  answer: "Answer",
};

export function useOrchestrate() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [diagram, setDiagram] = useState<DiagramState>({
    activeModel: null,
    activeTool: null,
    modelId: "",
    turnHistory: [],
  });
  const [trace, setTrace] = useState<TraceEntry[]>([]);
  const [totalCost, setTotalCost] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setDiagram({ activeModel: null, activeTool: null, modelId: "", turnHistory: [] });
    setTrace([]);
    setTotalCost(0);
    setIsRunning(false);
  }, []);

  const run = useCallback(async (question: string, maxTurns = 30) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsRunning(true);

    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", content: question },
    ]);
    setTrace([]);
    setTotalCost(0);
    setDiagram({ activeModel: "orchestrator", activeTool: null, modelId: "", turnHistory: [] });

    const thinkingId = nextId();

    try {
      const resp = await fetch("/api/orchestrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, max_turns: maxTurns }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "error", content: `Request failed: ${resp.status}` },
        ]);
        setIsRunning(false);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && eventType) {
            try {
              const data = JSON.parse(line.slice(6));
              const event: SSEEvent = { ...data, type: eventType };
              handleEvent(event, thinkingId);
            } catch {
              // skip malformed
            }
            eventType = "";
          }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "error", content: String(e) },
        ]);
      }
    } finally {
      setIsRunning(false);
      setDiagram((prev) => ({ ...prev, activeModel: null, activeTool: null, modelId: "" }));
    }
  }, []);

  function handleEvent(event: SSEEvent, thinkingId: string) {
    switch (event.type) {
      case "status":
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "status", content: event.message },
        ]);
        break;

      case "thinking":
        setMessages((prev) => {
          const existing = prev.findIndex((m) => m.id === thinkingId);
          const msg: ChatMessage = {
            id: thinkingId,
            role: "thinking",
            content: event.content,
            elapsed_s: event.elapsed_s,
          };
          if (existing >= 0) {
            const copy = [...prev];
            copy[existing] = msg;
            return copy;
          }
          return [...prev, msg];
        });
        setDiagram((prev) => ({ ...prev, activeModel: "orchestrator", activeTool: null }));
        break;

      case "tool_call":
        setMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== thinkingId);
          return [
            ...filtered,
            {
              id: nextId(),
              role: "tool_call",
              content: `${TOOL_LABELS[event.tool] ?? event.tool} via ${event.display_name}`,
              tool: event.tool,
              model_id: event.model_id,
              display_name: event.display_name,
              diagram_key: event.diagram_key,
              orch_latency_ms: event.orch_latency_ms,
            },
          ];
        });
        setDiagram((prev) => ({
          ...prev,
          activeModel: event.diagram_key,
          activeTool: event.tool,
          modelId: event.model_id,
        }));
        break;

      case "search_result":
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "search",
            content: event.count
              ? `Found ${event.count} results for: ${event.query}`
              : `No results found for: ${event.query}`,
            query: event.query,
            count: event.count,
            display_name: event.display_name,
            diagram_key: event.diagram_key,
            latency_ms: event.latency_ms,
            in_tokens: event.in_tokens,
            out_tokens: event.out_tokens,
          },
        ]);
        setTrace((prev) => [...prev, event]);
        setTotalCost(event.total_cost);
        setDiagram((prev) => ({
          ...prev,
          turnHistory: [
            ...prev.turnHistory,
            { turn: event.turn, tool: "search", display: event.display_name, latency_ms: event.latency_ms },
          ],
        }));
        break;

      case "reasoning_result":
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "reasoning",
            content: event.code_executed
              ? `Code executed successfully`
              : "Generated reasoning code",
            code_preview: event.code_preview,
            exec_output: event.exec_output,
            display_name: event.display_name,
            diagram_key: event.diagram_key,
            latency_ms: event.latency_ms,
            in_tokens: event.in_tokens,
            out_tokens: event.out_tokens,
          },
        ]);
        setTrace((prev) => [...prev, event]);
        setTotalCost(event.total_cost);
        setDiagram((prev) => ({
          ...prev,
          turnHistory: [
            ...prev.turnHistory,
            { turn: event.turn, tool: "enhance_reasoning", display: event.display_name, latency_ms: event.latency_ms },
          ],
        }));
        break;

      case "answer_attempt":
      case "answer_final":
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: event.is_final ? "answer_final" : "answer_attempt",
            content: event.prediction,
            prediction: event.prediction,
            is_final: event.is_final,
            display_name: event.display_name,
            diagram_key: event.diagram_key,
            latency_ms: event.latency_ms,
            in_tokens: event.in_tokens,
            out_tokens: event.out_tokens,
          },
        ]);
        setTrace((prev) => [...prev, event]);
        setTotalCost(event.total_cost);
        setDiagram((prev) => ({
          ...prev,
          turnHistory: [
            ...prev.turnHistory,
            { turn: event.turn, tool: "answer", display: event.display_name, latency_ms: event.latency_ms },
          ],
        }));
        break;

      case "done":
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "final",
            content: event.final_answer,
            total_turns: event.total_turns,
            total_tokens: event.total_tokens,
            total_cost: event.total_cost,
          },
        ]);
        if (event.tool_trace) setTrace(event.tool_trace);
        setTotalCost(event.total_cost);
        break;

      case "error":
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "error", content: event.message },
        ]);
        break;
    }
  }

  return { messages, diagram, trace, totalCost, isRunning, run, clear };
}
