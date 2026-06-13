import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useState } from "react";
import { apiJson } from "../api/watcher";
import { type ActorHistoryResponse, type PersonSuggestion } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";
import { posterUrl } from "../lib/poster";
import { PersonTypeahead } from "../components/PersonTypeahead";
import { useWatchlist } from "../contexts/WatchlistContext";

export function ActorPage() {
  const { isOnWatchlist, toggle, isToggling } = useWatchlist();
  const [selected, setSelected] = useState<PersonSuggestion | null>(null);
  const [result, setResult] = useState<ActorHistoryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function search(person: PersonSuggestion) {
    const name = person.name?.trim();
    if (!name) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const data = await apiJson<ActorHistoryResponse>(
        `/history/actor?name=${encodeURIComponent(name)}`,
      );
      setResult(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  function handleSelect(person: PersonSuggestion) {
    setSelected(person);
    setResult(null);
    setErr(null);
    void search(person);
  }

  function handleClear() {
    setSelected(null);
    setResult(null);
    setErr(null);
  }

  const profileSrc = result?.person.profile_path
    ? posterUrl(result.person.profile_path, "w185")
    : null;

  return (
    <div className="w-full">
      <h1 className="page-title">
        Actor Search
      </h1>
      <p className="text-muted mb-6">Find which titles in your history feature a specific actor or director.</p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5">
          <span className="field-label">
            Name
          </span>
          <PersonTypeahead
            selected={selected}
            onSelect={handleSelect}
            onClear={handleClear}
            placeholder="e.g. Cate Blanchett, Christopher Nolan…"
          />
        </label>
      </div>

      {err && <ErrorBox message={err} />}

      {busy && (
        <LoadingBox label="Searching your history…" />
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
            <p className="empty-state">
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
                    watchlistOn={item.media_type ? isOnWatchlist(item.id, item.media_type) : undefined}
                    watchlistLoading={item.media_type ? isToggling(item.id, item.media_type) : undefined}
                    onWatchlistToggle={
                      item.media_type
                        ? () => void toggle({ id: item.id, title: item.title ?? "", mediaType: item.media_type!, posterPath: item.poster_path })
                        : undefined
                    }
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
