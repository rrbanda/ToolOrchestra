import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";

const TOOL_COLORS: Record<string, string> = {
  search: "#3b82f6",
  enhance_reasoning: "#f97316",
  answer: "#22c55e",
};

const TOOL_ICONS: Record<string, string> = {
  search: "🔍",
  enhance_reasoning: "⚙️",
  answer: "✅",
};

function TokenBadge({ inTok, outTok, latencyMs }: { inTok?: number; outTok?: number; latencyMs?: number }) {
  if (!inTok && !outTok && !latencyMs) return null;
  const parts: string[] = [];
  if (latencyMs != null) parts.push(`${(latencyMs / 1000).toFixed(1)}s`);
  if (inTok != null && outTok != null) parts.push(`${inTok}+${outTok} tok`);
  return (
    <span className="text-[11px] text-white/30 font-mono">{parts.join(" · ")}</span>
  );
}

function ThinkingBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="animate-slide-in rounded-xl bg-indigo-500/10 border border-indigo-500/20 p-4 max-w-[90%]">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse-glow" />
        <span className="text-xs font-semibold text-indigo-300">
          Orchestrator reasoning... ({msg.elapsed_s}s)
        </span>
      </div>
      <p className="text-xs text-white/50 leading-relaxed font-mono whitespace-pre-wrap break-words">
        {msg.content.slice(-400)}
      </p>
    </div>
  );
}

function ToolCallCard({ msg }: { msg: ChatMessage }) {
  const color = TOOL_COLORS[msg.tool ?? ""] ?? "#6b7280";
  const icon = TOOL_ICONS[msg.tool ?? ""] ?? "🔧";
  return (
    <div
      className="animate-slide-in rounded-lg px-4 py-2.5 max-w-[90%] flex items-center gap-3"
      style={{ background: `${color}15`, borderLeft: `3px solid ${color}` }}
    >
      <span className="text-base">{icon}</span>
      <span className="text-sm font-medium text-white/80">{msg.content}</span>
      <span className="ml-auto text-[11px] text-white/30 font-mono">
        calling...
      </span>
    </div>
  );
}

function SearchResult({ msg }: { msg: ChatMessage }) {
  return (
    <div className="animate-slide-in rounded-xl bg-blue-500/8 border border-blue-500/15 p-4 max-w-[90%]">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm">🔍</span>
        <span className="text-sm font-semibold text-blue-300">Search</span>
        <span className="text-xs text-white/40">via {msg.display_name}</span>
      </div>
      <p className="text-sm text-white/70">
        {msg.count ? (
          <>Found <strong className="text-blue-300">{msg.count}</strong> results for: <em className="text-white/50">{msg.query}</em></>
        ) : (
          <>No results found for: <em className="text-white/50">{msg.query}</em></>
        )}
      </p>
      <div className="mt-2 pt-2 border-t border-white/5">
        <TokenBadge inTok={msg.in_tokens} outTok={msg.out_tokens} latencyMs={msg.latency_ms} />
      </div>
    </div>
  );
}

function ReasoningResult({ msg }: { msg: ChatMessage }) {
  return (
    <div className="animate-slide-in rounded-xl bg-orange-500/8 border border-orange-500/15 p-4 max-w-[90%]">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm">⚙️</span>
        <span className="text-sm font-semibold text-orange-300">Reasoning</span>
        <span className="text-xs text-white/40">via {msg.display_name}</span>
      </div>
      {msg.code_preview && (
        <pre className="mt-2 p-3 rounded-lg bg-black/30 text-xs text-green-300 font-mono overflow-x-auto max-h-40 whitespace-pre-wrap break-words">
          {msg.code_preview}
        </pre>
      )}
      {msg.exec_output && (
        <div className="mt-2 p-2 rounded bg-black/20 text-xs text-white/60 font-mono">
          Output: {msg.exec_output}
        </div>
      )}
      <div className="mt-2 pt-2 border-t border-white/5">
        <TokenBadge inTok={msg.in_tokens} outTok={msg.out_tokens} latencyMs={msg.latency_ms} />
      </div>
    </div>
  );
}

function AnswerCard({ msg }: { msg: ChatMessage }) {
  const isFinal = msg.role === "answer_final";
  return (
    <div
      className={`animate-slide-in rounded-xl p-4 max-w-[90%] ${
        isFinal
          ? "bg-green-500/10 border border-green-500/25"
          : "bg-green-500/5 border border-green-500/10"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm">{isFinal ? "✅" : "💡"}</span>
        <span className={`text-sm font-semibold ${isFinal ? "text-green-300" : "text-green-400/70"}`}>
          {isFinal ? "Answer (verified)" : "Answer attempt"}
        </span>
        <span className="text-xs text-white/40">via {msg.display_name}</span>
      </div>
      <p className={`text-sm leading-relaxed ${isFinal ? "text-white/90" : "text-white/60"}`}>
        {msg.prediction ?? msg.content}
      </p>
      {!isFinal && (
        <p className="mt-2 text-xs text-white/30 italic">Orchestrator will verify or retry...</p>
      )}
      <div className="mt-2 pt-2 border-t border-white/5">
        <TokenBadge inTok={msg.in_tokens} outTok={msg.out_tokens} latencyMs={msg.latency_ms} />
      </div>
    </div>
  );
}

function FinalSummary({ msg }: { msg: ChatMessage }) {
  return (
    <div className="animate-slide-in rounded-xl bg-gradient-to-br from-green-500/12 to-emerald-500/8 border border-green-500/20 p-5 max-w-[90%]">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🎯</span>
        <span className="text-sm font-bold text-green-300">Final Answer</span>
      </div>
      <p className="text-base text-white/95 font-medium leading-relaxed">{msg.content}</p>
      <div className="mt-3 pt-3 border-t border-white/10 flex gap-4 flex-wrap">
        {msg.total_turns != null && (
          <span className="text-xs text-white/40">{msg.total_turns} turns</span>
        )}
        {msg.total_tokens != null && (
          <span className="text-xs text-white/40">{msg.total_tokens.toLocaleString()} tokens</span>
        )}
        {msg.total_cost != null && (
          <span className="text-xs text-white/40">${msg.total_cost.toFixed(4)}</span>
        )}
      </div>
    </div>
  );
}

function UserMessage({ msg }: { msg: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="rounded-xl bg-indigo-600/20 border border-indigo-500/20 px-4 py-3 max-w-[75%]">
        <p className="text-sm text-white/90">{msg.content}</p>
      </div>
    </div>
  );
}

function StatusMessage({ msg }: { msg: ChatMessage }) {
  return (
    <div className="flex items-center gap-2 px-2">
      <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
      <span className="text-xs text-white/40">{msg.content}</span>
    </div>
  );
}

function ErrorMessage({ msg }: { msg: ChatMessage }) {
  return (
    <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-2.5 max-w-[90%]">
      <span className="text-sm text-red-300">{msg.content}</span>
    </div>
  );
}

function MessageRenderer({ msg }: { msg: ChatMessage }) {
  switch (msg.role) {
    case "user":
      return <UserMessage msg={msg} />;
    case "thinking":
      return <ThinkingBubble msg={msg} />;
    case "tool_call":
      return <ToolCallCard msg={msg} />;
    case "search":
      return <SearchResult msg={msg} />;
    case "reasoning":
      return <ReasoningResult msg={msg} />;
    case "answer_attempt":
    case "answer_final":
      return <AnswerCard msg={msg} />;
    case "final":
      return <FinalSummary msg={msg} />;
    case "status":
      return <StatusMessage msg={msg} />;
    case "error":
      return <ErrorMessage msg={msg} />;
    default:
      return null;
  }
}

interface ChatPanelProps {
  messages: ChatMessage[];
}

export default function ChatPanel({ messages }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="text-4xl mb-4">🎼</div>
          <h2 className="text-lg font-semibold text-white/80 mb-2">ToolOrchestra</h2>
          <p className="text-sm text-white/40 leading-relaxed">
            Ask a question below or pick a preset. The Orchestrator-8B will route
            it through specialist models — you'll see each step here and the
            routing diagram will update live.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-3">
      {messages.map((msg) => (
        <div key={msg.id}>
          <MessageRenderer msg={msg} />
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
