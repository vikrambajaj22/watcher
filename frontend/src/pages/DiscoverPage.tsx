import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useState } from "react";
import { apiFetch } from "../api/client";
import { type DescribeFilters, type DiscoverItem } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";

type MediaTypeFilter = "both" | "movie" | "tv";

export function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [mediaType, setMediaType] = useState<MediaTypeFilter>("both");
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
        body: JSON.stringify({ query: q, limit: 20, media_type: mediaType }),
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
      <h1 className="page-title">
        Discover
      </h1>
      <p className="text-muted mb-6">
        Describe what you want to watch — mood, genre, era, cast — and we'll find it.
      </p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="field-label">
            What are you in the mood for?
          </span>
          <input
            className="glass-input rounded-lg text-text px-3 py-2.5 text-[16px] sm:text-sm w-full"
            type="text"
            placeholder="e.g. 90s sci-fi with practical effects, feel-good comedy like The Grand Budapest Hotel…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void search(); }}
          />
        </label>
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="field-label">
            Type
          </span>
          <select
            className="glass-input rounded-lg text-text px-3 py-2.5 text-[16px] sm:text-sm w-36"
            value={mediaType}
            onChange={(e) => setMediaType(e.target.value as MediaTypeFilter)}
          >
            <option value="both">Movies & TV</option>
            <option value="movie">Movies only</option>
            <option value="tv">TV only</option>
          </select>
        </label>
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !query.trim()}
          onClick={() => void search()}
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </div>

      {err && <ErrorBox message={err} />}

      {busy && (
        <LoadingBox label="Searching…" />
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
            <p className="empty-state">No results found. Try a different description.</p>
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
