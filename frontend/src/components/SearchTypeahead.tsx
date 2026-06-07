import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../api/client";
import { type SearchHit } from "../api/watcher";
import { posterUrl, placeholderPoster } from "../lib/poster";

type Props = {
  onSelect: (hit: SearchHit) => void;
  selected: SearchHit | null;
  onClear: () => void;
  placeholder?: string;
};

export function SearchTypeahead({
  onSelect,
  selected,
  onClear,
  placeholder = "Search title…",
}: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [focusedIdx, setFocusedIdx] = useState(-1);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  async function doSearch(q: string) {
    if (!q.trim()) {
      setHits([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    try {
      const r = await apiFetch(`/search?q=${encodeURIComponent(q.trim())}&limit=6`);
      if (!r.ok) { setHits([]); return; }
      const data = (await r.json()) as { results: SearchHit[] };
      setHits(data.results ?? []);
      setOpen(true);
      setFocusedIdx(-1);
    } catch {
      setHits([]);
    } finally {
      setLoading(false);
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => void doSearch(val), 300);
  }

  function pick(hit: SearchHit) {
    setQuery("");
    setHits([]);
    setOpen(false);
    onSelect(hit);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || hits.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusedIdx((i) => Math.min(i + 1, hits.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && focusedIdx >= 0) {
      e.preventDefault();
      pick(hits[focusedIdx]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  if (selected) {
    return (
      <div className="typeahead-selected">
        <img
          className="typeahead-selected-poster"
          src={posterUrl(selected.poster_path, "w92") ?? placeholderPoster(selected.title)}
          alt=""
        />
        <div className="typeahead-selected-info">
          <div className="typeahead-selected-title">{selected.title}</div>
          <div className="typeahead-option-meta">
            {selected.media_type === "tv" ? "TV" : "Movie"}
            {selected.year ? ` · ${selected.year}` : ""}
          </div>
        </div>
        <button type="button" className="typeahead-clear" onClick={onClear} aria-label="Clear">
          ✕
        </button>
      </div>
    );
  }

  return (
    <div className="typeahead-wrap" ref={wrapRef}>
      <div className="typeahead-input-wrap">
        <input
          className="input typeahead-input"
          value={query}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={() => hits.length > 0 && setOpen(true)}
          placeholder={placeholder}
          autoComplete="off"
        />
        {loading && <span className="typeahead-spinner" aria-hidden />}
      </div>
      {open && hits.length > 0 && (
        <ul className="typeahead-dropdown" role="listbox">
          {hits.map((hit, idx) => (
            <li
              key={`${hit.id}-${hit.media_type}`}
              className={`typeahead-option${idx === focusedIdx ? " focused" : ""}`}
              role="option"
              aria-selected={idx === focusedIdx}
              onMouseDown={() => pick(hit)}
            >
              <img
                className="typeahead-option-poster"
                src={posterUrl(hit.poster_path, "w92") ?? placeholderPoster(hit.title)}
                alt=""
              />
              <div className="typeahead-option-info">
                <div className="typeahead-option-title">{hit.title}</div>
                <div className="typeahead-option-meta">
                  {hit.media_type === "tv" ? "TV" : "Movie"}
                  {hit.year ? ` · ${hit.year}` : ""}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
