import { useEffect, useState } from "react";
import { type TasteProfile, apiJson } from "../api/watcher";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: TasteProfile }
  | { status: "err"; message: string };

function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span className={`px-2.5 py-1 rounded-full text-sm font-medium border ${color}`}>
      {label}
    </span>
  );
}

function Section({ title, items, color }: { title: string; items: string[]; color: string }) {
  if (!items.length) return null;
  return (
    <div>
      <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-muted mb-2">{title}</p>
      <div className="flex flex-wrap gap-2">
        {items.map((item) => (
          <Pill key={item} label={item} color={color} />
        ))}
      </div>
    </div>
  );
}

export function TasteProfilePage() {
  const [state, setState] = useState<State>({ status: "idle" });

  useEffect(() => {
    setState({ status: "loading" });
    apiJson<TasteProfile>("/taste-profile")
      .then((data) => setState({ status: "ok", data }))
      .catch((e: unknown) =>
        setState({ status: "err", message: e instanceof Error ? e.message : "Request failed" })
      );
  }, []);

  return (
    <div className="w-full max-w-2xl">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">
        Taste Profile
      </h1>
      <p className="text-muted mb-6">
        An AI read of your viewing habits, based on your watch history.
      </p>

      {state.status === "loading" && (
        <div className="flex items-center gap-4 p-5 glass rounded-2xl" role="status" aria-live="polite">
          <div className="size-7 rounded-full border-[3px] border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
          <p className="text-sm m-0">Analyzing your history…</p>
        </div>
      )}

      {state.status === "err" && (
        <div className="p-4 bg-surface border border-danger/40 rounded-xl">
          <strong className="text-danger">Error: </strong>{state.message}
        </div>
      )}

      {state.status === "ok" && (
        <div className="flex flex-col gap-4">
          {/* Signature */}
          <div className="p-5 glass rounded-2xl">
            <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-muted mb-2">Your taste in a nutshell</p>
            <p className="text-lg font-semibold tracking-[-0.02em] text-text m-0">
              {state.data.signature}
            </p>
          </div>

          {/* Summary */}
          <div className="p-5 glass rounded-2xl">
            <p className="text-[0.7rem] font-bold uppercase tracking-[0.08em] text-muted mb-2">Overview</p>
            <p className="text-sm leading-relaxed text-text/85 m-0">{state.data.summary}</p>
          </div>

          {/* Genres + Themes + Avoid */}
          <div className="p-5 glass rounded-2xl flex flex-col gap-5">
            <Section
              title="Genres you gravitate toward"
              items={state.data.genres}
              color="bg-accent/10 text-accent border-accent/20"
            />
            <Section
              title="Recurring themes"
              items={state.data.themes}
              color="bg-accent/10 text-accent border-accent/20"
            />
            <Section
              title="Rarely watches"
              items={state.data.avoid}
              color="bg-white/5 text-muted border-border"
            />
          </div>

          <p className="text-xs text-muted text-right">
            Based on {state.data.history_count} titles · refreshes hourly
          </p>
        </div>
      )}
    </div>
  );
}
