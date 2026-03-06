import type { TraceEntry } from "../types";

const TOOL_COLORS: Record<string, string> = {
  search: "#3b82f6",
  enhance_reasoning: "#f97316",
  answer: "#22c55e",
};

interface Props {
  trace: TraceEntry[];
  totalCost: number;
}

export default function ToolTrace({ trace, totalCost }: Props) {
  if (trace.length === 0) {
    return (
      <div className="text-center py-6 text-white/30 text-[13px]">
        Waiting for orchestration...
      </div>
    );
  }

  return (
    <div className="space-y-1.5 text-[12px]">
      {trace.map((t, i) => {
        const color = TOOL_COLORS[t.tool] ?? "#6b7280";
        let detail = "";
        if (t.query) detail = `query: ${t.query.slice(0, 50)}`;
        else if (t.code_executed) detail = "code executed";
        else if (t.note) detail = t.note;

        return (
          <div
            key={i}
            className="flex items-center gap-2.5 px-3 py-2 rounded-r-lg bg-white/[0.03] animate-slide-in"
            style={{ borderLeft: `3px solid ${color}` }}
          >
            <span className="font-bold text-white/70 min-w-[28px]">T{t.turn}</span>
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
            <span className="text-white/60 min-w-[90px]">{t.tool}</span>
            <span className="text-white/40 flex-1 truncate">{t.specialist ?? ""}</span>
            {t.latency_ms != null && (
              <span className="text-white/30 font-mono tabular-nums">{t.latency_ms}ms</span>
            )}
            {t.in_tokens != null && t.out_tokens != null && (
              <span className="text-white/25 font-mono tabular-nums min-w-[72px] text-right">
                {t.in_tokens}+{t.out_tokens}
              </span>
            )}
            {t.est_cost_usd != null && (
              <span className="text-white/20 font-mono tabular-nums min-w-[70px] text-right">
                ${t.est_cost_usd.toFixed(6)}
              </span>
            )}
            {detail && (
              <span className="hidden xl:block text-white/25 truncate max-w-[120px]" title={detail}>
                {detail}
              </span>
            )}
          </div>
        );
      })}

      <div className="text-right text-[13px] font-semibold text-white/40 pt-2">
        Total cost: ${totalCost.toFixed(6)}
      </div>
    </div>
  );
}
