const TMDB_IMG = "https://image.tmdb.org/t/p";

export function posterUrl(
  posterPath: string | null | undefined,
  size: "w92" | "w154" | "w185" | "w342" | "w500" | "w780" = "w342",
): string | null {
  if (!posterPath) return null;
  return `${TMDB_IMG}/${size}${posterPath}`;
}

export function placeholderPoster(text: string): string {
  const t = encodeURIComponent(text.slice(0, 40) || "—");
  return `https://via.placeholder.com/342x513/2a2e38/9aa0a8?text=${t}`;
}
