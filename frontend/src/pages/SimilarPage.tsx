import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type SimilarResponse, type SimilarResult } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { SimilarResultRow } from "../components/SimilarResultRow";

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
    <div className="page page-wide">
      <h1 className="page-title">Similar Titles</h1>
      <p className="lede">Find movies or shows similar to a title via TMDB.</p>

      <div className="card form-card">
        <label className="field field-block">
          <span className="field-label">Title</span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => { setSelectedHit(hit); setResults(null); setErr(null); }}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </label>
        <div className="toolbar-row">
          <label className="field">
            <span className="field-label">Results</span>
            <input
              className="input input-narrow"
              type="number"
              min={1}
              max={40}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </label>
          <label className="field checkbox-field">
            <input
              type="checkbox"
              checked={crossType}
              onChange={(e) => setCrossType(e.target.checked)}
            />
            <span>Cross-type</span>
          </label>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy || !selectedHit}
          onClick={() => void search()}
        >
          {busy ? "Searching…" : "Find Similar"}
        </button>
      </div>

      {err && <div className="card card-error">{err}</div>}

      {busy && (
        <div className="card loading-panel" role="status" aria-live="polite">
          <div className="loading-spinner" aria-hidden />
          <p className="loading-panel-text">Finding Similar Titles…</p>
        </div>
      )}

      {!busy && results !== null && (
        <>
          {sourceLabel && (
            <h2 className="section-title">Similar To: {sourceLabel}</h2>
          )}
          {results.length === 0 ? (
            <p className="muted card empty-results-note">No similar titles found.</p>
          ) : (
            <div className="history-card-list">
              {results.map((item, index) => (
                <SimilarResultRow
                  key={`${item.id}-${item.media_type}-${index}`}
                  item={item}
                  rank={index + 1}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
