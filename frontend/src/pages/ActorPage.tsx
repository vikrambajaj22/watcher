import { useState } from "react";
import { apiJson } from "../api/watcher";
import { type ActorHistoryResponse } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";
import { posterUrl } from "../lib/poster";

export function ActorPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<ActorHistoryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function search() {
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const data = await apiJson<ActorHistoryResponse>(
        `/history/actor?name=${encodeURIComponent(q)}`,
      );
      setResult(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const profileSrc = result?.person.profile_path
    ? posterUrl(result.person.profile_path, "w185")
    : null;

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">
        Actor Search
      </h1>
      <p className="text-muted mb-6">Find which titles in your history feature a specific actor or director.</p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
            Name
          </span>
          <input
            className="glass-input rounded-lg text-text px-3 py-2.5 text-sm w-full"
            type="text"
            placeholder="e.g. Cate Blanchett, Christopher Nolan…"
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
          {busy ? "Searching…" : "Search History"}
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
          <p className="text-sm m-0">Searching your history…</p>
        </div>
      )}

      {!busy && result && (
        <>
          <div className="flex items-center gap-4 p-4 glass rounded-2xl mb-5">
            {profileSrc ? (
              <img
                src={profileSrc}
                alt=""
                loading="lazy"
                className="size-14 rounded-full object-cover shrink-0"
              />
            ) : (
              <div className="size-14 rounded-full bg-surface border border-border shrink-0 flex items-center justify-center text-xl text-muted">
                {result.person.name?.[0] ?? "?"}
              </div>
            )}
            <div>
              <h2 className="text-lg font-semibold tracking-tight m-0">{result.person.name}</h2>
              {result.person.known_for_department && (
                <p className="text-sm text-muted m-0">{result.person.known_for_department}</p>
              )}
            </div>
          </div>

          {result.items.length === 0 ? (
            <p className="text-muted p-5 glass rounded-2xl">
              {result.person.name} doesn't appear in your watch history.
            </p>
          ) : (
            <>
              <p className="text-sm text-muted mb-3">
                Found in <span className="text-text font-medium">{result.items.length}</span>{" "}
                {result.items.length === 1 ? "title" : "titles"} you've watched
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {result.items.map((item) => (
                  <MediaCard
                    key={`${item.id}-${item.media_type}`}
                    id={item.id}
                    title={item.title ?? "Unknown"}
                    mediaType={item.media_type}
                    posterPath={item.poster_path}
                    subtitle={item.character ?? undefined}
                    similarLink
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
