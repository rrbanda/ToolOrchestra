import { useState, useRef, useEffect } from "react";

const PRESET_CHIPS: Array<{ label: string; question: string }> = [
  { label: "Math + Primality", question: "What is 17 factorial divided by 13 factorial, and is the result a prime number?" },
  { label: "Population Compare", question: "What is the population of Tokyo compared to Paris?" },
  { label: "Calculus", question: "Solve: If f(x) = 3x^2 + 2x - 5, find f'(x) and evaluate f'(3)." },
  { label: "Nobel Prize 2023", question: "Who won the Nobel Prize in Physics in 2023 and what was it for?" },
  { label: "Compound Interest", question: "Calculate the compound interest on $10,000 at 5% annual rate compounded quarterly for 3 years." },
];

interface Props {
  onSubmit: (question: string) => void;
  onClear: () => void;
  isRunning: boolean;
}

export default function InputBar({ onSubmit, onClear, isRunning }: Props) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isRunning) inputRef.current?.focus();
  }, [isRunning]);

  const submit = () => {
    const q = value.trim();
    if (!q || isRunning) return;
    onSubmit(q);
    setValue("");
  };

  return (
    <div className="border-t border-white/10 bg-surface-1/80 backdrop-blur-sm p-3">
      {/* Preset chips */}
      <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
        {PRESET_CHIPS.map((chip) => (
          <button
            key={chip.label}
            onClick={() => {
              if (isRunning) return;
              setValue(chip.question);
              inputRef.current?.focus();
            }}
            disabled={isRunning}
            className="flex-shrink-0 px-3 py-1.5 text-xs text-white/60 bg-white/[0.04] border border-white/10 
                       rounded-lg hover:bg-indigo-500/12 hover:border-indigo-500/30 hover:text-white/80
                       transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Input row */}
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Ask anything — math, factual, reasoning..."
          disabled={isRunning}
          className="flex-1 px-4 py-2.5 bg-surface-2 border border-white/10 rounded-xl text-sm text-white/90
                     placeholder:text-white/30 focus:outline-none focus:border-indigo-500/50 focus:ring-1 
                     focus:ring-indigo-500/20 transition-all duration-200 disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={isRunning || !value.trim()}
          className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium 
                     rounded-xl transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed
                     min-w-[56px]"
        >
          {isRunning ? (
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            </span>
          ) : (
            "Ask"
          )}
        </button>
        <button
          onClick={onClear}
          className="px-3 py-2.5 text-sm text-white/50 bg-white/[0.04] border border-white/10 
                     rounded-xl hover:bg-white/[0.08] hover:text-white/70 transition-all duration-200"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
