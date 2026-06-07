import { useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type WillLikeResponse } from "../api/watcher";
import { AiBlurb } from "../components/AiBlurb";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { VerdictBadge } from "../components/VerdictBadge";
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
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Will I Like It?</h1>
      <p className="text-muted mb-6">
        A short model-assisted read on whether a title fits your taste, based on your watch history.
      </p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
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
          className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-sm cursor-pointer transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_24px_-4px_rgba(74,222,128,0.45)] disabled:opacity-50 disabled:cursor-not-allowed border-0"
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
        <div className="flex items-center gap-4 p-5 glass rounded-2xl mb-4" role="status" aria-live="polite">
          <div className="size-7 rounded-full border-[3px] border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
          <p className="text-sm m-0">Checking Your Taste Fit…</p>
        </div>
      )}

      {!busy && result && (
        <article className="flex items-start gap-4 sm:gap-5 p-4 sm:p-5 glass rounded-2xl">
          <div className="w-[64px] sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30">
            <img
              src={posterUrl(result.item.poster_path, "w185") ?? placeholderPoster(displayTitle)}
              alt=""
              loading="lazy"
              className="w-full aspect-[2/3] object-cover block"
            />
          </div>
          <div className="flex-1 min-w-0">
            {/* Type badge */}
            <span
              className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-2 py-0.5 rounded-full mb-1.5 ${
                result.already_watched
                  ? "bg-emerald-400/10 text-emerald-300"
                  : isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
              }`}
            >
              {result.already_watched ? "Watched" : isTv ? "TV" : "Film"}
            </span>

            <h2 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-2">
              {displayTitle}
            </h2>

            {/* Verdict — prominent, in the reading flow */}
            {!result.already_watched && (
              <div className="mb-3">
                <VerdictBadge willLike={result.will_like} score={result.score} />
              </div>
            )}
            {result.already_watched && (
              <p className="text-sm text-muted mb-3">Already in your history.</p>
            )}

            <AiBlurb>{result.explanation}</AiBlurb>
          </div>

          <div className="shrink-0 self-start mt-0.5">
            {itemId != null && itemId > 0 && result.item.media_type && (
              <Link
                className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 transition-all whitespace-nowrap"
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
