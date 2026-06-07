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
      <div className="flex items-center gap-3 px-3 py-2 glass rounded-lg min-h-[2.35rem]">
        <img
          className="w-[26px] h-[39px] rounded object-cover bg-bg shrink-0"
          src={posterUrl(selected.poster_path, "w92") ?? placeholderPoster(selected.title)}
          alt=""
        />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold truncate">{selected.title}</div>
          <div className="text-xs text-muted mt-0.5">
            {selected.media_type === "tv" ? "TV" : "Movie"}
            {selected.year ? ` · ${selected.year}` : ""}
          </div>
        </div>
        <button
          type="button"
          onClick={onClear}
          aria-label="Clear"
          className="text-muted hover:text-text text-sm cursor-pointer bg-transparent border-0 px-1 shrink-0"
        >
          ✕
        </button>
      </div>
    );
  }

  return (
    <div className="relative w-full" ref={wrapRef}>
      <div className="relative flex items-center">
        <input
          className="w-full glass-input rounded-lg text-text px-2.5 py-2 text-[16px] sm:text-sm pr-8"
          value={query}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={() => hits.length > 0 && setOpen(true)}
          placeholder={placeholder}
          autoComplete="off"
        />
        {loading && (
          <span
            className="absolute right-2.5 size-4 rounded-full border-2 border-border border-t-accent animate-spin [animation-duration:0.7s]"
            aria-hidden
          />
        )}
      </div>
      {open && hits.length > 0 && (
        <ul
          className="absolute top-[calc(100%+4px)] left-0 right-0 glass-dark rounded-2xl shadow-2xl shadow-black/50 z-[200] overflow-hidden list-none m-0 p-0"
          role="listbox"
        >
          {hits.map((hit, idx) => (
            <li
              key={`${hit.id}-${hit.media_type}`}
              className={`flex items-center gap-3 px-3 py-2 cursor-pointer border-b border-border last:border-b-0 transition-colors ${
                idx === focusedIdx ? "bg-white/10" : "hover:bg-white/7"
              }`}
              role="option"
              aria-selected={idx === focusedIdx}
              onMouseDown={() => pick(hit)}
            >
              <img
                className="w-[30px] h-[45px] rounded object-cover bg-bg shrink-0"
                src={posterUrl(hit.poster_path, "w92") ?? placeholderPoster(hit.title)}
                alt=""
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold truncate">{hit.title}</div>
                <div className="text-xs text-muted mt-0.5">
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
