import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../api/client";
import { type PersonSuggestion } from "../api/watcher";
import { posterUrl } from "../lib/poster";

type Props = {
  onSelect: (person: PersonSuggestion) => void;
  selected: PersonSuggestion | null;
  onClear: () => void;
  placeholder?: string;
};

const PLACEHOLDER_PERSON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='30' height='45' viewBox='0 0 30 45'%3E%3Crect width='30' height='45' fill='%23333'/%3E%3Ccircle cx='15' cy='16' r='7' fill='%23666'/%3E%3Cellipse cx='15' cy='36' rx='11' ry='8' fill='%23666'/%3E%3C/svg%3E";

export function PersonTypeahead({
  onSelect,
  selected,
  onClear,
  placeholder = "Search actor or director…",
}: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<PersonSuggestion[]>([]);
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
      const r = await apiFetch(`/search/person?q=${encodeURIComponent(q.trim())}&limit=6`);
      if (!r.ok) { setHits([]); return; }
      const data = (await r.json()) as { results: PersonSuggestion[] };
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

  function pick(person: PersonSuggestion) {
    setQuery("");
    setHits([]);
    setOpen(false);
    onSelect(person);
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
          className="size-8 rounded-full object-cover bg-surface shrink-0"
          src={selected.profile_path ? posterUrl(selected.profile_path, "w185") ?? PLACEHOLDER_PERSON : PLACEHOLDER_PERSON}
          alt=""
        />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold truncate">{selected.name}</div>
          {selected.known_for_department && (
            <div className="text-xs text-muted mt-0.5">{selected.known_for_department}</div>
          )}
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
          className="absolute top-[calc(100%+4px)] left-0 right-0 glass-popup rounded-2xl z-[200] overflow-hidden list-none m-0 p-0"
          role="listbox"
        >
          {hits.map((person, idx) => (
            <li
              key={person.id}
              className={`flex items-center gap-3 px-3 py-2 cursor-pointer border-b border-border last:border-b-0 transition-colors ${
                idx === focusedIdx ? "bg-white/10" : "hover:bg-white/7"
              }`}
              role="option"
              aria-selected={idx === focusedIdx}
              onMouseDown={() => pick(person)}
            >
              <img
                className="size-9 rounded-full object-cover bg-surface shrink-0"
                src={person.profile_path ? posterUrl(person.profile_path, "w185") ?? PLACEHOLDER_PERSON : PLACEHOLDER_PERSON}
                alt=""
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold truncate">{person.name}</div>
                <div className="text-xs text-muted mt-0.5 truncate">
                  {person.known_for_department && (
                    <span>{person.known_for_department}</span>
                  )}
                  {person.known_for_department && person.known_for && person.known_for.length > 0 && (
                    <span> · </span>
                  )}
                  {person.known_for && person.known_for.length > 0 && (
                    <span>{person.known_for.join(", ")}</span>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
