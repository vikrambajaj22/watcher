import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type SimilarResponse, type SimilarResult } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { SimilarResultRow } from "../components/SimilarResultRow";

type Mode = "search" | "history";
type HistoryHit = { id?: number; tmdb_id?: number; title?: string; name?: string; media_type?: string };

const inputCls = "glass-input rounded-lg text-text px-2.5 py-2 text-[16px] sm:text-sm";

export function SimilarPage() {
  const [searchParams] = useSearchParams();
  const [mode, setMode] = useState<Mode>("search");
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [k, setK] = useState(20);
  const [crossType, setCrossType] = useState(false);

  const [results, setResults] = useState<SimilarResult[] | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // History mode state
  const [historyItems, setHistoryItems] = useState<HistoryHit[] | null>(null);
  const [historySearch, setHistorySearch] = useState("");
  const [historyBusy, setHistoryBusy] = useState(false);
  const historyInputRef = useRef<HTMLInputElement>(null);

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

  async function searchFromHistory(item: HistoryHit) {
    const id = item.tmdb_id ?? item.id;
    const mt = item.media_type;
    if (!id || (mt !== "movie" && mt !== "tv")) { setErr("Invalid item"); return; }
    setSourceLabel(item.title ?? item.name ?? "");
    await callSimilar({ tmdb_id: id, media_type: mt, k: kRef.current, cross_type: crossType });
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

  useEffect(() => {
    if (mode !== "history" || historyItems !== null) return;
    setHistoryBusy(true);
    apiFetch("/history?include_posters=false")
      .then(r => r.json() as Promise<HistoryHit[]>)
      .then(d => setHistoryItems(Array.isArray(d) ? d : []))
      .catch(() => setHistoryItems([]))
      .finally(() => setHistoryBusy(false));
  }, [mode, historyItems]);

  useEffect(() => {
    if (mode === "history") setTimeout(() => historyInputRef.current?.focus(), 50);
  }, [mode]);

  const filteredHistory = useMemo(() => {
    if (!historyItems) return [];
    const q = historySearch.toLowerCase().trim();
    if (!q) return [];
    return historyItems
      .filter(h => (h.title ?? h.name ?? "").toLowerCase().includes(q))
      .slice(0, 20);
  }, [historyItems, historySearch]);

  return (
    <div className="w-full">
      <h1 className="page-title">Similar Titles</h1>
      <p className="text-muted mb-6">Find movies or shows similar to a title via TMDB.</p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        {/* Mode toggle */}
        <div className="flex gap-1 p-1 glass rounded-xl w-fit mb-5">
          {(["search", "history"] as Mode[]).map(m => (
            <button
              key={m}
              type="button"
              onClick={() => { setMode(m); setResults(null); setErr(null); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer border-0 font-sans ${
                mode === m ? "bg-accent/20 text-accent" : "text-muted hover:text-text bg-transparent"
              }`}
            >
              {m === "search" ? "By Title" : "From History"}
            </button>
          ))}
        </div>

        {mode === "search" ? (
          <label className="flex flex-col gap-1.5 mb-4">
            <span className="field-label">
              Title
            </span>
            <SearchTypeahead
              selected={selectedHit}
              onSelect={(hit) => { setSelectedHit(hit); setResults(null); setErr(null); }}
              onClear={() => setSelectedHit(null)}
              placeholder="Search movie or show…"
            />
          </label>
        ) : (
          <div className="flex flex-col gap-2 mb-4">
            <span className="field-label">
              Pick from history
            </span>
            <input
              ref={historyInputRef}
              className={`${inputCls} w-full`}
              type="text"
              placeholder="Filter titles…"
              value={historySearch}
              onChange={e => setHistorySearch(e.target.value)}
            />
            {historyBusy && (
              <p className="text-xs text-muted">Loading history…</p>
            )}
            {!historyBusy && historyItems !== null && historySearch.trim() !== "" && (
              <div className="flex flex-col max-h-48 overflow-y-auto rounded-lg border border-border/30 divide-y divide-border/20">
                {filteredHistory.length === 0 ? (
                  <p className="text-xs text-muted px-3 py-2">No matches</p>
                ) : filteredHistory.map((h, i) => {
                  const title = h.title ?? h.name ?? "Unknown";
                  const mt = h.media_type ?? "";
                  return (
                    <button
                      key={`${h.id ?? h.tmdb_id}-${i}`}
                      type="button"
                      onClick={() => void searchFromHistory(h)}
                      className="flex items-center justify-between gap-2 px-3 py-2 text-left text-sm hover:bg-white/5 transition-colors cursor-pointer bg-transparent border-0 font-sans text-text"
                    >
                      <span className="truncate">{title}</span>
                      <span className="text-[0.65rem] uppercase tracking-wide text-muted shrink-0">{mt}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-3 items-end mb-4">
          <label className="flex flex-col gap-1">
            <span className="field-label">
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

        {mode === "search" && (
          <button
            type="button"
            className="btn-primary"
            disabled={busy || !selectedHit}
            onClick={() => void search()}
          >
            {busy ? "Searching…" : "Find Similar"}
          </button>
        )}
      </div>

      {err && <ErrorBox message={err} />}

      {busy && (
        <LoadingBox label="Finding Similar Titles…" />
      )}

      {!busy && results !== null && (
        <>
          {sourceLabel && (
            <h2 className="text-lg font-semibold tracking-tight mb-3">
              Similar to: <span className="text-accent">{sourceLabel}</span>
            </h2>
          )}
          {results.length === 0 ? (
            <p className="empty-state">
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
