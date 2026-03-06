import { useOrchestrate } from "./hooks/useOrchestrate";
import ChatPanel from "./components/ChatPanel";
import RoutingDiagram from "./components/RoutingDiagram";
import ToolTrace from "./components/ToolTrace";
import InputBar from "./components/InputBar";

export default function App() {
  const { messages, diagram, trace, totalCost, isRunning, run, clear } = useOrchestrate();

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-white/10 bg-surface-1/60 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xl">🎼</span>
          <div>
            <h1 className="text-base font-bold text-white/90 tracking-tight">ToolOrchestra</h1>
            <p className="text-[11px] text-white/35">Multi-Model Orchestration Demo</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isRunning && (
            <span className="flex items-center gap-1.5 text-xs text-indigo-300">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
              Processing
            </span>
          )}
          <span className="text-[11px] text-white/25">
            Orchestrator-8B · Llama · Qwen · Gemini
          </span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat column */}
        <div className="flex-1 flex flex-col min-w-0">
          <ChatPanel messages={messages} />
          <InputBar onSubmit={run} onClear={clear} isRunning={isRunning} />
        </div>

        {/* Side panel */}
        <aside className="w-[380px] xl:w-[420px] border-l border-white/10 bg-surface-1/40 flex flex-col flex-shrink-0 overflow-y-auto">
          {/* Model Routing */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-sm font-semibold text-white/70 mb-3">Model Routing</h2>
            <RoutingDiagram state={diagram} />
          </div>

          {/* Tool Trace */}
          <div className="p-4 flex-1 overflow-y-auto">
            <h2 className="text-sm font-semibold text-white/70 mb-3">Tool Trace</h2>
            <ToolTrace trace={trace} totalCost={totalCost} />
          </div>
        </aside>
      </div>
    </div>
  );
}
