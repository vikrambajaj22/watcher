import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { placeholderPoster, posterUrl } from "../lib/poster";

type Props = {
  id: number;
  title: string;
  mediaType?: string;
  posterPath?: string | null;
  subtitle?: string;
  overview?: string | null;
  footer?: ReactNode;
  similarLink?: boolean;
};

export function MediaCard({
  id,
  title,
  mediaType,
  posterPath,
  subtitle,
  overview,
  footer,
  similarLink,
}: Props) {
  const src = posterUrl(posterPath ?? null, "w342") ?? placeholderPoster(title);
  const mt = (mediaType || "movie").toLowerCase();
  const isTv = mt === "tv";

  return (
    <article className="media-card">
      <div className="media-card-poster-wrap">
        <img className="media-card-poster" src={src} alt="" loading="lazy" />
      </div>
      <div className="media-card-body">
        {mediaType && (
          <span className={`history-card-kind ${isTv ? "tv" : "movie"}`}>
            {isTv ? "TV" : "Film"}
          </span>
        )}
        <h3 className="media-card-title">{title}</h3>
        {subtitle && <p className="media-card-reasoning">{subtitle}</p>}
        {overview && String(overview).trim() && (
          <p className="media-card-sub muted media-card-overview">
            {String(overview).slice(0, 280)}
            {String(overview).length > 280 ? "…" : ""}
          </p>
        )}
        {footer ?? null}
        {similarLink && (
          <Link
            className="btn btn-secondary history-card-link media-card-similar"
            to={`/similar?id=${id}&type=${encodeURIComponent(mt)}`}
          >
            Find Similar
          </Link>
        )}
      </div>
    </article>
  );
}
