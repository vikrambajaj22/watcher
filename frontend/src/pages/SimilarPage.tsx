import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type SimilarResponse, type SimilarResult } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { SimilarResultRow } from "../components/SimilarResultRow";

const inputCls = "glass-input rounded-lg text-text px-2.5 py-2 text-sm";

export function SimilarPage() {
  const [searchParams] = useSearchParams();
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [k, setK] = useState(20);
  const [crossType, setCrossType] = useState(false);

  const [results, setResults] = useState<SimilarResult[] | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const kRef = useRef(k);
  kRef.current = k;

  async function callSimilar(payload: Record<string, unknown>) {
    setBusy(true);
    setErr(null);
    setResults(null);
    try {
      const r = await apiFetch("/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const raw = await r.text();
      let j: unknown;
      try { j = JSON.parse(raw); } catch { j = { detail: raw }; }
      if (!r.ok) {
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : raw,
        );
        return;
      }
      const resp = j as SimilarResponse;
      if (resp.source_title) setSourceLabel(resp.source_title);
      setResults(resp.results ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  async function search() {
    if (!selectedHit) { setErr("Select a title first"); return; }
    setSourceLabel(selectedHit.title);
    await callSimilar({ tmdb_id: selectedHit.id, media_type: selectedHit.media_type, k, cross_type: crossType });
  }

  useEffect(() => {
    const id = searchParams.get("id");
    const type = searchParams.get("type");
    if (!id || (type !== "movie" && type !== "tv")) return;
    const n = Number(id);
    if (!Number.isFinite(n) || n < 1) return;
    void callSimilar({ tmdb_id: n, media_type: type, k: kRef.current, cross_type: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- URL-driven search only
  }, [searchParams]);

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Similar Titles</h1>
      <p className="text-muted mb-6">Find movies or shows similar to a title via TMDB.</p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
            Title
          </span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => { setSelectedHit(hit); setResults(null); setErr(null); }}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </label>
        <div className="flex flex-wrap gap-3 items-end mb-4">
          <label className="flex flex-col gap-1">
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
              Results
            </span>
            <input
              className={`${inputCls} w-20`}
              type="number"
              min={1}
              max={40}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </label>
          <label className="flex items-center gap-2 self-end pb-2">
            <input
              type="checkbox"
              checked={crossType}
              onChange={(e) => setCrossType(e.target.checked)}
            />
            <span className="text-sm">Cross-type</span>
          </label>
        </div>
        <button
          type="button"
          className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-sm cursor-pointer transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_24px_-4px_rgba(74,222,128,0.45)] disabled:opacity-50 disabled:cursor-not-allowed border-0"
          disabled={busy || !selectedHit}
          onClick={() => void search()}
        >
          {busy ? "Searching…" : "Find Similar"}
        </button>
      </div>

      {err && (
        <div className="p-4 glass border-danger/40 rounded-xl mb-4">
          <strong className="text-danger">Error: </strong>
          {err}
        </div>
      )}

      {busy && (
        <div className="flex items-center gap-4 p-5 glass rounded-2xl mb-4" role="status" aria-live="polite">
          <div className="size-7 rounded-full border-[3px] border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
          <p className="text-sm m-0">Finding Similar Titles…</p>
        </div>
      )}

      {!busy && results !== null && (
        <>
          {sourceLabel && (
            <h2 className="text-lg font-semibold tracking-tight mb-3">
              Similar to: <span className="text-accent">{sourceLabel}</span>
            </h2>
          )}
          {results.length === 0 ? (
            <p className="text-muted p-5 glass rounded-2xl">
              No similar titles found.
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              {results.map((item, index) => (
                <SimilarResultRow
                  key={`${item.id}-${item.media_type}-${index}`}
                  item={item}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
