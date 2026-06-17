import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type SimilarResponse, type SimilarResult } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { SimilarResultRow } from "../components/SimilarResultRow";
import { placeholderPoster, posterUrl } from "../lib/poster";

type Mode = "search" | "history" | "to-history";
type HistoryHit = { id?: number; tmdb_id?: number; title?: string; name?: string; media_type?: string; poster_path?: string | null };

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
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historySelected, setHistorySelected] = useState<HistoryHit | null>(null);
  const [historyFocusedIdx, setHistoryFocusedIdx] = useState(-1);
  const historyInputRef = useRef<HTMLInputElement>(null);
  const historyWrapRef = useRef<HTMLDivElement>(null);

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

  async function searchToHistory() {
    if (!selectedHit) { setErr("Select a title first"); return; }
    setSourceLabel(selectedHit.title);
    await callSimilar({ tmdb_id: selectedHit.id, media_type: selectedHit.media_type, k, cross_type: crossType, filter_to_history: true });
  }

  async function searchFromHistory(item: HistoryHit) {
    const id = item.tmdb_id ?? item.id;
    const mt = item.media_type;
    if (!id || (mt !== "movie" && mt !== "tv")) { setErr("Invalid item"); return; }
    setHistorySelected(item);
    setHistoryOpen(false);
    setHistorySearch("");
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
    apiFetch("/history?include_posters=true")
      .then(r => r.json() as Promise<HistoryHit[]>)
      .then(d => setHistoryItems(Array.isArray(d) ? d : []))
      .catch(() => setHistoryItems([]))
      .finally(() => setHistoryBusy(false));
  }, [mode, historyItems]);

  useEffect(() => {
    if (mode === "history") setTimeout(() => historyInputRef.current?.focus(), 50);
    setHistorySelected(null);
    setHistorySearch("");
    setHistoryOpen(false);
  }, [mode]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (historyWrapRef.current && !historyWrapRef.current.contains(e.target as Node)) {
        setHistoryOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

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

      <div className="relative z-10 p-5 glass-dark rounded-2xl mb-4">
        {/* Mode toggle */}
        <div className="flex gap-1 p-1 glass rounded-xl w-fit mb-5">
          {(["search", "history", "to-history"] as Mode[]).map(m => (
            <button
              key={m}
              type="button"
              onClick={() => { setMode(m); setResults(null); setErr(null); setSelectedHit(null); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer border-0 font-sans ${
                mode === m ? "bg-accent/20 text-accent" : "text-muted hover:text-text bg-transparent"
              }`}
            >
              {m === "search" ? "By Title" : m === "history" ? "From History" : "To History"}
            </button>
          ))}
        </div>

        {mode === "search" || mode === "to-history" ? (
          <label className="flex flex-col gap-1.5 mb-4">
            <span className="field-label">Title</span>
            <SearchTypeahead
              selected={selectedHit}
              onSelect={(hit) => { setSelectedHit(hit); setResults(null); setErr(null); }}
              onClear={() => setSelectedHit(null)}
              placeholder="Search movie or show…"
            />
          </label>
        ) : (
          <div className="flex flex-col gap-2 mb-4">
            <span className="field-label">Pick from history</span>
            <div className="relative w-full" ref={historyWrapRef}>
              {historySelected ? (
                <div className="flex items-center gap-3 px-3 py-2 glass rounded-lg min-h-[2.35rem]">
                  <img
                    className="w-[26px] h-[39px] rounded object-contain bg-bg shrink-0"
                    src={posterUrl(historySelected.poster_path ?? null, "w92") ?? placeholderPoster()}
                    alt=""
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold truncate">{historySelected.title ?? historySelected.name}</div>
                    <div className="text-xs text-muted mt-0.5 uppercase tracking-wide">{historySelected.media_type}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => { setHistorySelected(null); setResults(null); setErr(null); setTimeout(() => historyInputRef.current?.focus(), 50); }}
                    aria-label="Clear"
                    className="text-muted hover:text-text text-sm cursor-pointer bg-transparent border-0 px-1 shrink-0"
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="relative flex items-center">
                  <input
                    ref={historyInputRef}
                    className={`${inputCls} w-full pr-8`}
                    type="text"
                    placeholder="Filter titles…"
                    value={historySearch}
                    onChange={e => { setHistorySearch(e.target.value); setHistoryOpen(true); setHistoryFocusedIdx(-1); }}
                    onFocus={() => historySearch.trim() && setHistoryOpen(true)}
                    onKeyDown={e => {
                      if (!historyOpen || filteredHistory.length === 0) return;
                      if (e.key === "ArrowDown") { e.preventDefault(); setHistoryFocusedIdx(i => Math.min(i + 1, filteredHistory.length - 1)); }
                      else if (e.key === "ArrowUp") { e.preventDefault(); setHistoryFocusedIdx(i => Math.max(i - 1, 0)); }
                      else if (e.key === "Enter" && historyFocusedIdx >= 0) { e.preventDefault(); void searchFromHistory(filteredHistory[historyFocusedIdx]); }
                      else if (e.key === "Escape") setHistoryOpen(false);
                    }}
                    autoComplete="off"
                  />
                  {historyBusy && (
                    <span className="absolute right-2.5 size-4 rounded-full border-2 border-border border-t-accent animate-spin [animation-duration:0.7s]" aria-hidden />
                  )}
                </div>
              )}
              {!historySelected && historyOpen && filteredHistory.length > 0 && (
                <ul className="absolute top-[calc(100%+4px)] left-0 right-0 glass-popup rounded-2xl z-[200] overflow-hidden list-none m-0 p-0 max-h-60 overflow-y-auto">
                  {filteredHistory.map((h, idx) => {
                    const title = h.title ?? h.name ?? "Unknown";
                    const mt = h.media_type ?? "";
                    return (
                      <li
                        key={`${h.id ?? h.tmdb_id}-${idx}`}
                        className={`flex items-center gap-3 px-3 py-2 cursor-pointer border-b border-border last:border-b-0 transition-colors ${
                          idx === historyFocusedIdx ? "bg-white/10" : "hover:bg-white/7"
                        }`}
                        onMouseDown={() => void searchFromHistory(h)}
                      >
                        <img
                          className="w-[30px] h-[45px] rounded object-contain bg-bg shrink-0"
                          src={posterUrl(h.poster_path ?? null, "w92") ?? placeholderPoster()}
                          alt=""
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm truncate">{title}</div>
                          <div className="text-xs text-muted mt-0.5 uppercase tracking-wide">{mt}</div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
              {!historySelected && historyOpen && historySearch.trim() !== "" && !historyBusy && filteredHistory.length === 0 && (
                <div className="absolute top-[calc(100%+4px)] left-0 right-0 glass-popup rounded-2xl z-[200] px-3 py-2.5">
                  <p className="text-xs text-muted m-0">No matches</p>
                </div>
              )}
            </div>
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

        {(mode === "search" || mode === "to-history") && (
          <button
            type="button"
            className="btn-primary"
            disabled={busy || !selectedHit}
            onClick={() => void (mode === "to-history" ? searchToHistory() : search())}
          >
            {busy ? "Searching…" : mode === "to-history" ? "Find in History" : "Find Similar"}
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
              {mode === "to-history" ? "From your history, similar to: " : "Similar to: "}
              <span className="text-accent">{sourceLabel}</span>
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
                  hideWillLike={mode === "to-history"}
                  hideWatchlist={mode === "to-history"}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
