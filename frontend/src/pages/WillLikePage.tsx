import { useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type WillLikeResponse } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { posterUrl, placeholderPoster } from "../lib/poster";

export function WillLikePage() {
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [result, setResult] = useState<WillLikeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!selectedHit) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await apiFetch("/will-like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tmdb_id: selectedHit.id, media_type: selectedHit.media_type }),
      });
      const text = await r.text();
      let j: unknown;
      try { j = JSON.parse(text); } catch { j = { detail: text }; }
      if (!r.ok) {
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : text,
        );
        return;
      }
      setResult(j as WillLikeResponse);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const displayTitle = result?.item.title ?? result?.item.name ?? "—";
  const itemId = result?.item.id;
  const itemMt = (result?.item.media_type ?? "movie").toLowerCase();
  const isTv = itemMt === "tv";

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.03em] mb-1.5">Will I Like It?</h1>
      <p className="text-muted max-w-[52ch] mb-6">
        A short model-assisted read on whether a title fits your taste, based on your watch history.
      </p>

      <div className="p-5 bg-surface border border-border rounded-xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
            Title
          </span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => { setSelectedHit(hit); setResult(null); setErr(null); }}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </label>
        <button
          type="button"
          className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-white font-semibold text-sm cursor-pointer transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed border-0"
          disabled={busy || !selectedHit}
          onClick={() => void submit()}
        >
          {busy ? "Checking…" : "Check"}
        </button>
      </div>

      {err && (
        <div className="p-4 bg-surface border border-danger/40 rounded-xl mb-4">
          <strong className="text-danger">Error: </strong>
          {err}
        </div>
      )}

      {busy && (
        <div className="flex items-center gap-4 p-5 bg-surface border border-border rounded-xl mb-4" role="status" aria-live="polite">
          <div className="size-7 rounded-full border-[3px] border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
          <p className="text-sm m-0">Checking Your Taste Fit…</p>
        </div>
      )}

      {!busy && result && (
        <article className="flex gap-4 sm:gap-5 p-4 sm:p-5 bg-surface border border-border rounded-xl">
          <div className="w-[64px] sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30">
            <img
              src={posterUrl(result.item.poster_path, "w185") ?? placeholderPoster(displayTitle)}
              alt=""
              loading="lazy"
              className="w-full aspect-[2/3] object-cover block"
            />
          </div>
          <div className="flex-1 min-w-0">
            {result.already_watched ? (
              <span className="inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded mb-1.5 bg-emerald-400/10 text-emerald-300">
                Watched
              </span>
            ) : (
              <span
                className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded mb-1.5 ${
                  isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
                }`}
              >
                {isTv ? "TV" : "Film"}
              </span>
            )}
            <h2 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-1">
              {displayTitle}
            </h2>
            <p className="text-sm text-muted mb-2">
              {result.already_watched
                ? "Already in your history."
                : `Score: ${(result.score * 100).toFixed(0)}%`}
            </p>
            <p className="text-sm italic text-muted leading-relaxed m-0">{result.explanation}</p>
          </div>
          <div className="hidden sm:flex flex-col gap-1.5 text-xs shrink-0 w-28">
            <div className="flex justify-between items-baseline gap-2">
              <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Verdict</span>
              <span className="font-semibold">
                {result.already_watched ? (
                  <span className="px-1.5 py-0.5 rounded text-[0.8rem] font-semibold bg-emerald-400/15 text-emerald-300">
                    Seen
                  </span>
                ) : result.will_like ? (
                  <span className="px-1.5 py-0.5 rounded text-[0.8rem] font-semibold bg-emerald-400/15 text-emerald-300">
                    Likely Yes
                  </span>
                ) : (
                  <span className="px-1.5 py-0.5 rounded text-[0.8rem] font-semibold bg-yellow-400/12 text-yellow-300">
                    Probably Not
                  </span>
                )}
              </span>
            </div>
          </div>
          <div className="shrink-0 self-center">
            {itemId != null && itemId > 0 && result.item.media_type && (
              <Link
                className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 hover:no-underline transition-all whitespace-nowrap"
                to={`/similar?id=${itemId}&type=${encodeURIComponent(itemMt)}`}
              >
                Find Similar
              </Link>
            )}
          </div>
        </article>
      )}
    </div>
  );
}
