type Props = {
  id: number;
  mediaType?: string;
  linkTo?: "tmdb" | "trakt";
  traktSlug?: string | null;
};

/** Branded outbound link to a title on TMDB or Trakt. */
export function ExternalMediaLink({ id, mediaType, linkTo = "tmdb", traktSlug }: Props) {
  if (!id || id <= 0) return null;
  const isTv = (mediaType || "movie").toLowerCase() === "tv";
  const isTrakt = linkTo === "trakt";
  // Prefer Trakt's canonical slug URL; fall back to its TMDB-id search redirect.
  const traktHref = traktSlug
    ? `https://trakt.tv/${isTv ? "shows" : "movies"}/${traktSlug}`
    : `https://trakt.tv/search/tmdb/${id}?id_type=${isTv ? "show" : "movie"}`;
  const href = isTrakt
    ? traktHref
    : `https://www.themoviedb.org/${isTv ? "tv" : "movie"}/${id}`;

  return (
    <a
      className="inline-flex items-center shrink-0 hover:no-underline hover:opacity-80 transition-opacity"
      href={href}
      target="_blank"
      rel="noreferrer"
      aria-label={isTrakt ? "Trakt" : "TMDB"}
    >
      <span
        className="text-[8px] font-bold leading-none tracking-tight px-1 py-0.5 rounded"
        style={
          isTrakt
            ? { background: "#ed1c24", color: "#fff" }
            : { background: "linear-gradient(90deg,#90cea1,#01b4e4)", color: "#032541" }
        }
      >
        {isTrakt ? "TRAKT" : "TMDB"}
      </span>
    </a>
  );
}
