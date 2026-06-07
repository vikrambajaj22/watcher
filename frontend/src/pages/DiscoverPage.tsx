import { useState } from "react";
import { apiFetch } from "../api/client";
import { type DescribeFilters, type DiscoverItem } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";

export function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<DiscoverItem[] | null>(null);
  const [filters, setFilters] = useState<DescribeFilters | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function search() {
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    setErr(null);
    setResults(null);
    setFilters(null);
    try {
      const r = await apiFetch("/discover/describe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, limit: 20 }),
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
      const resp = j as { results: DiscoverItem[]; filters?: DescribeFilters };
      setResults(resp.results ?? []);
      setFilters(resp.filters ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const filterChips: string[] = [];
  if (filters) {
    if (filters.genres?.length) filterChips.push(...filters.genres);
    if (filters.cast?.length) filterChips.push(...filters.cast.map((n) => `with ${n}`));
    if (filters.keywords?.length) filterChips.push(...filters.keywords);
    if (filters.year_from && filters.year_to) filterChips.push(`${filters.year_from}–${filters.year_to}`);
    else if (filters.year_from) filterChips.push(`from ${filters.year_from}`);
    else if (filters.year_to) filterChips.push(`up to ${filters.year_to}`);
  }

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">
        Discover
      </h1>
      <p className="text-muted mb-6">
        Describe what you want to watch — mood, genre, era, cast — and we'll find it.
      </p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
            What are you in the mood for?
          </span>
          <input
            className="glass-input rounded-lg text-text px-3 py-2.5 text-sm w-full"
            type="text"
            placeholder="e.g. 90s sci-fi with practical effects, feel-good comedy like The Grand Budapest Hotel…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void search(); }}
          />
        </label>
        <button
          type="button"
          className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-sm cursor-pointer transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_24px_-4px_rgba(74,222,128,0.45)] disabled:opacity-50 disabled:cursor-not-allowed border-0"
          disabled={busy || !query.trim()}
          onClick={() => void search()}
        >
          {busy ? "Searching…" : "Search"}
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
          <p className="text-sm m-0">Searching…</p>
        </div>
      )}

      {!busy && results !== null && (
        <>
          {filterChips.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {filterChips.map((chip) => (
                <span
                  key={chip}
                  className="text-[0.7rem] font-semibold uppercase tracking-[0.06em] px-2 py-0.5 rounded-full bg-accent/10 text-accent border border-accent/20"
                >
                  {chip}
                </span>
              ))}
            </div>
          )}
          {results.length === 0 ? (
            <p className="text-muted p-5 glass rounded-2xl">No results found. Try a different description.</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {results.map((item) => (
                <MediaCard
                  key={`${item.id}-${item.media_type}`}
                  id={item.id}
                  title={item.title ?? "Unknown"}
                  mediaType={item.media_type}
                  posterPath={item.poster_path}
                  overview={item.overview}
                  watched={item.watched}
                  similarLink
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
