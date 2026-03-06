import type { DiagramState } from "../types";

const TOOL_COLORS: Record<string, string> = {
  search: "#3b82f6",
  enhance_reasoning: "#f97316",
  answer: "#22c55e",
};

const TOOL_LABELS: Record<string, string> = {
  search: "Search",
  enhance_reasoning: "Reasoning",
  answer: "Answer",
};

interface ModelBoxProps {
  name: string;
  role: string;
  infra: string;
  color: string;
  bgColor: string;
  active: boolean;
  glowColor?: string;
}

function ModelBox({ name, role, infra, color, bgColor, active, glowColor }: ModelBoxProps) {
  return (
    <div
      className="rounded-xl border-2 px-3 py-2.5 text-center transition-all duration-300"
      style={{
        borderColor: active ? (glowColor ?? color) : color,
        background: bgColor,
        boxShadow: active ? `0 0 20px ${glowColor ?? color}66` : "none",
      }}
    >
      <div className="text-[13px] font-bold text-gray-100 whitespace-nowrap">{name}</div>
      <div className="text-[10px] text-white/55 whitespace-nowrap">{role}</div>
      <div className="text-[9px] text-white/35 mt-0.5 whitespace-nowrap">{infra}</div>
    </div>
  );
}

interface Props {
  state: DiagramState;
}

export default function RoutingDiagram({ state }: Props) {
  const { activeModel, activeTool, modelId, turnHistory } = state;
  const color = TOOL_COLORS[activeTool ?? ""] ?? "#6b7280";
  const toolLabel = TOOL_LABELS[activeTool ?? ""] ?? "";
  const isSpecialistActive = activeModel === "llama" || activeModel === "qwen" || activeModel === "gemini";

  const ds = "rgba(255,255,255,0.18)";
  const ts = isSpecialistActive ? color : ds;
  const tw = isSpecialistActive ? 3 : 2;

  const branchStroke = (key: string) => activeModel === key ? color : ds;
  const branchWidth = (key: string) => activeModel === key ? 3 : 2;

  const dotCx = activeModel === "llama" ? 100 : activeModel === "qwen" ? 300 : activeModel === "gemini" ? 500 : null;
  const colIdx = activeModel === "llama" ? 0 : activeModel === "qwen" ? 1 : activeModel === "gemini" ? 2 : -1;

  return (
    <div className="select-none">
      {/* Orchestrator box */}
      <div className="flex justify-center mb-0">
        <ModelBox
          name="Orchestrator-8B"
          role="RL-Trained Agent"
          infra="GPU 1 · L40S 48 GB"
          color="#818cf8"
          bgColor="rgba(99,102,241,0.18)"
          active={activeModel === "orchestrator"}
          glowColor="#a5b4fc"
        />
      </div>

      {/* SVG connectors */}
      <svg viewBox="0 0 600 82" preserveAspectRatio="xMidYMid meet" className="w-full block">
        {/* trunk */}
        <line x1={300} y1={0} x2={300} y2={25} stroke={ts} strokeWidth={tw} />
        {/* horizontal bar */}
        <line x1={100} y1={25} x2={500} y2={25} stroke={ds} strokeWidth={2} />
        {/* left branch */}
        <line x1={100} y1={25} x2={100} y2={68} stroke={branchStroke("llama")} strokeWidth={branchWidth("llama")} />
        <polygon points="94,68 100,78 106,68" fill={branchStroke("llama")} />
        {/* centre branch */}
        <line x1={300} y1={25} x2={300} y2={68} stroke={branchStroke("qwen")} strokeWidth={branchWidth("qwen")} />
        <polygon points="294,68 300,78 306,68" fill={branchStroke("qwen")} />
        {/* right branch */}
        <line x1={500} y1={25} x2={500} y2={68} stroke={branchStroke("gemini")} strokeWidth={branchWidth("gemini")} />
        <polygon points="494,68 500,78 506,68" fill={branchStroke("gemini")} />

        {/* animated dot */}
        {dotCx != null && (
          <circle cx={dotCx} r={4} fill={color} opacity={0}>
            <animate attributeName="cy" from="25" to="68" dur="0.8s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0;1;0" dur="0.8s" repeatCount="indefinite" />
          </circle>
        )}
      </svg>

      {/* Specialist grid */}
      <div className="grid grid-cols-3 gap-0">
        <div className="flex justify-center px-1">
          <div className="w-full">
            <ModelBox
              name="Llama 3.2"
              role="Search · Reasoning"
              infra="GPU 2 · L4 24 GB"
              color="#22d3ee"
              bgColor="rgba(6,182,212,0.14)"
              active={activeModel === "llama"}
              glowColor={color}
            />
          </div>
        </div>
        <div className="flex justify-center px-1">
          <div className="w-full">
            <ModelBox
              name="Qwen Math"
              role="Math Specialist"
              infra="GPU 3 · L4 24 GB"
              color="#c084fc"
              bgColor="rgba(168,85,247,0.14)"
              active={activeModel === "qwen"}
              glowColor={color}
            />
          </div>
        </div>
        <div className="flex justify-center px-1">
          <div className="w-full">
            <ModelBox
              name="Gemini"
              role="2.5 Pro · Flash"
              infra="Cloud (LlamaStack)"
              color="#fbbf24"
              bgColor="rgba(245,158,11,0.14)"
              active={activeModel === "gemini"}
              glowColor={color}
            />
          </div>
        </div>
      </div>

      {/* Active label */}
      {activeTool && isSpecialistActive && colIdx >= 0 && (
        <div className="grid grid-cols-3 gap-0 mt-1.5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex justify-center">
              {i === colIdx && (
                <span
                  className="text-[10px] font-semibold px-2 py-0.5 rounded-md border inline-block"
                  style={{
                    color,
                    background: "rgba(255,255,255,0.06)",
                    borderColor: "rgba(255,255,255,0.12)",
                  }}
                >
                  {toolLabel} {modelId}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Turn history */}
      {turnHistory.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/10 space-y-1">
          {turnHistory.slice(-8).map((h, i) => {
            const hColor = TOOL_COLORS[h.tool] ?? "#6b7280";
            return (
              <div key={i} className="flex items-center gap-2 text-[11px] text-white/50">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: hColor }} />
                <span className="font-semibold text-white/70 w-6">T{h.turn}</span>
                <span className="w-28">{TOOL_LABELS[h.tool] ?? h.tool}</span>
                <span className="text-white/35 flex-1">{h.display}</span>
                <span className="text-white/30 font-mono tabular-nums">{h.latency_ms}ms</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
