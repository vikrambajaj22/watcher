import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiFetch } from "../api/client";
import {
  apiJson,
  type KnnResponse,
  type KnnResult,
} from "../api/watcher";
import { SimilarResultRow } from "../components/SimilarResultRow";

type Tab = "title" | "text" | "id";

export function SimilarPage() {
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>("title");

  const [titleIn, setTitleIn] = useState("");
  const [titleInputMedia, setTitleInputMedia] = useState<"movie" | "tv">("movie");
  const [titleResultsMedia, setTitleResultsMedia] = useState<
    "movie" | "tv" | "all"
  >("all");
  const [kTitle, setKTitle] = useState(10);

  const [textQuery, setTextQuery] = useState("");
  const [textResultsMedia, setTextResultsMedia] = useState<
    "movie" | "tv" | "all"
  >("all");
  const [kText, setKText] = useState(10);

  const [tmdbId, setTmdbId] = useState(550);
  const [idInputMedia, setIdInputMedia] = useState<"movie" | "tv">("movie");
  const [idResultsMedia, setIdResultsMedia] = useState<
    "movie" | "tv" | "all"
  >("all");
  const [kId, setKId] = useState(10);

  const [results, setResults] = useState<KnnResult[] | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [disambig, setDisambig] = useState(false);
  const [lastPayload, setLastPayload] = useState<Record<string, unknown> | null>(
    null,
  );

  const idSearchOptsRef = useRef({ results: idResultsMedia, k: kId });
  idSearchOptsRef.current = { results: idResultsMedia, k: kId };

  async function callKnn(payload: Record<string, unknown>) {
    setBusy(true);
    setErr(null);
    setDisambig(false);
    setResults(null);
    setLastPayload(payload);
    try {
      const r = await apiFetch("/mcp/knn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const raw = await r.text();
      let j: unknown;
      try {
        j = JSON.parse(raw);
      } catch {
        j = { detail: raw };
      }
      if (r.status === 400) {
        const detail = JSON.stringify(j);
        if (
          detail.includes("input_media_type") ||
          detail.toLowerCase().includes("ambiguous")
        ) {
          setDisambig(true);
          return;
        }
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : raw,
        );
        setResults(null);
        return;
      }
      if (!r.ok) {
        setErr(raw);
        setResults(null);
        return;
      }
      const knn = j as KnnResponse;
      setResults(knn.results ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
      setResults(null);
    } finally {
      setBusy(false);
    }
  }

  async function runSearchById(
    idNum: number,
    inputMedia: "movie" | "tv",
    resultsMedia: "movie" | "tv" | "all",
    k: number,
  ) {
    setSourceLabel(`TMDB ${idNum} (${inputMedia})`);
    try {
      const path =
        `/admin/tmdb/${idNum}` +
        (inputMedia ? `?media_type=${inputMedia}` : "");
      const meta = await apiJson<Record<string, unknown>[]>(path);
      const first = meta[0];
      if (first) {
        setSourceLabel(
          String(first.title ?? first.name ?? `TMDB ${idNum}`),
        );
      }
    } catch {
      /* metadata optional */
    }
    await callKnn({
      tmdb_id: idNum,
      input_media_type: inputMedia,
      results_media_type: resultsMedia,
      k,
    });
  }

  useEffect(() => {
    const id = searchParams.get("id");
    const type = searchParams.get("type");
    if (!id || (type !== "movie" && type !== "tv")) return;
    const n = Number(id);
    if (!Number.isFinite(n) || n < 1) return;
    setTab("id");
    setTmdbId(n);
    setIdInputMedia(type);
    const { results, k } = idSearchOptsRef.current;
    void runSearchById(n, type, results, k);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- URL-driven search only; uses ref for k/results
  }, [searchParams]);

  async function searchByTitle() {
    if (!titleIn.trim()) {
      setErr("Enter A Title");
      return;
    }
    setSourceLabel(titleIn.trim());
    await callKnn({
      title: titleIn.trim(),
      input_media_type: titleInputMedia,
      results_media_type: titleResultsMedia,
      k: kTitle,
    });
  }

  async function searchByText() {
    if (!textQuery.trim()) {
      setErr("Enter A Description");
      return;
    }
    setSourceLabel(textQuery.slice(0, 80) + (textQuery.length > 80 ? "…" : ""));
    await callKnn({
      text: textQuery.trim(),
      results_media_type: textResultsMedia,
      k: kText,
    });
  }

  async function searchById() {
    await runSearchById(tmdbId, idInputMedia, idResultsMedia, kId);
  }

  function retryDisambig(choice: "movie" | "tv") {
    if (!lastPayload) return;
    void callKnn({ ...lastPayload, input_media_type: choice });
  }

  return (
    <div className="page page-wide">
      <h1 className="page-title">Similar Titles</h1>
      <p className="lede">
        Find titles similar to a film, show, free-text description, or TMDB id,
        using your library’s similarity index.
      </p>

      <div className="tabs">
        {(
          [
            ["title", "By Title"],
            ["text", "By Description"],
            ["id", "By TMDB ID"],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            type="button"
            className={tab === k ? "tab active" : "tab"}
            onClick={() => setTab(k)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "title" && (
        <div className="card form-card">
          <label className="field field-block">
            <span className="field-label">Title</span>
            <input
              className="input"
              value={titleIn}
              onChange={(e) => setTitleIn(e.target.value)}
            />
          </label>
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">Title Type</span>
              <select
                className="input"
                value={titleInputMedia}
                onChange={(e) =>
                  setTitleInputMedia(e.target.value as "movie" | "tv")
                }
              >
                <option value="movie">Movie</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label">Results</span>
              <select
                className="input"
                value={titleResultsMedia}
                onChange={(e) =>
                  setTitleResultsMedia(e.target.value as "movie" | "tv" | "all")
                }
              >
                <option value="all">All</option>
                <option value="movie">Movies</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label">K</span>
              <input
                className="input input-narrow"
                type="number"
                min={1}
                max={50}
                value={kTitle}
                onChange={(e) => setKTitle(Number(e.target.value))}
              />
            </label>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void searchByTitle()}
          >
            {busy ? "Searching…" : "Find Similar"}
          </button>
        </div>
      )}

      {tab === "text" && (
        <div className="card form-card">
          <label className="field field-block">
            <span className="field-label">Description</span>
            <textarea
              className="input textarea"
              rows={4}
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="e.g. slow-burn sci-fi with strong characters"
            />
          </label>
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">Results</span>
              <select
                className="input"
                value={textResultsMedia}
                onChange={(e) =>
                  setTextResultsMedia(e.target.value as "movie" | "tv" | "all")
                }
              >
                <option value="all">All</option>
                <option value="movie">Movies</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label">K</span>
              <input
                className="input input-narrow"
                type="number"
                min={1}
                max={50}
                value={kText}
                onChange={(e) => setKText(Number(e.target.value))}
              />
            </label>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void searchByText()}
          >
            {busy ? "Searching…" : "Find Similar"}
          </button>
        </div>
      )}

      {tab === "id" && (
        <div className="card form-card">
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">TMDB ID</span>
              <input
                className="input input-narrow"
                type="number"
                min={1}
                value={tmdbId}
                onChange={(e) => setTmdbId(Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span className="field-label">ID Type</span>
              <select
                className="input"
                value={idInputMedia}
                onChange={(e) =>
                  setIdInputMedia(e.target.value as "movie" | "tv")
                }
              >
                <option value="movie">Movie</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label">Results</span>
              <select
                className="input"
                value={idResultsMedia}
                onChange={(e) =>
                  setIdResultsMedia(e.target.value as "movie" | "tv" | "all")
                }
              >
                <option value="all">All</option>
                <option value="movie">Movies</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label">K</span>
              <input
                className="input input-narrow"
                type="number"
                min={1}
                max={50}
                value={kId}
                onChange={(e) => setKId(Number(e.target.value))}
              />
            </label>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void searchById()}
          >
            {busy ? "Searching…" : "Find Similar"}
          </button>
        </div>
      )}

      {disambig && lastPayload && (
        <div className="card card-warn">
          <p>Ambiguous Match — Choose Input Type:</p>
          <div className="actions">
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => retryDisambig("movie")}
            >
              Movie
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => retryDisambig("tv")}
            >
              TV
            </button>
          </div>
        </div>
      )}

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
            <p className="muted card empty-results-note">
              No similar titles found. Try another query or ask for more results
              above.
            </p>
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
