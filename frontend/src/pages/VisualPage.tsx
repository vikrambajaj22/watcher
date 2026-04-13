import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import type { ClusterItem, ClustersResponse } from "../api/watcher";
import { ClusterChart } from "../components/ClusterChart";
import { placeholderPoster, posterUrl } from "../lib/poster";

export function VisualPage() {
  const [media, setMedia] = useState<"all" | "movie" | "tv">("all");
  const [nClusters, setNClusters] = useState(6);
  const [data, setData] = useState<ClustersResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [openCluster, setOpenCluster] = useState<string | null>(null);

  const itemsByCluster = useMemo(() => {
    if (!data?.items.length) return new Map<string, ClusterItem[]>();
    const m = new Map<string, ClusterItem[]>();
    for (const it of data.items) {
      const k = String(it.cluster);
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(it);
    }
    return m;
  }, [data]);

  async function load() {
    setBusy(true);
    setErr(null);
    try {
      const p = new URLSearchParams();
      p.set("n_clusters", String(nClusters));
      if (media !== "all") p.set("media_type", media);
      const r = await apiFetch(`/visualize/clusters?${p.toString()}`);
      if (!r.ok) {
        setErr(await r.text());
        setData(null);
        return;
      }
      setData((await r.json()) as ClustersResponse);
      setOpenCluster(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page-wide">
      <h1 className="page-title">Visual Explorer</h1>
      <p className="lede">
        Your watch history laid out as clusters you can expand — browse posters
        and jump to similar titles.
      </p>

      <div className="toolbar card">
        <label className="field">
          <span className="field-label">Media</span>
          <select
            className="input"
            value={media}
            onChange={(e) => setMedia(e.target.value as typeof media)}
          >
            <option value="all">All</option>
            <option value="movie">Movies</option>
            <option value="tv">TV</option>
          </select>
        </label>
        <label className="field">
          <span className="field-label">Clusters</span>
          <input
            className="input input-narrow"
            type="number"
            min={3}
            max={15}
            value={nClusters}
            onChange={(e) => setNClusters(Number(e.target.value))}
          />
        </label>
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => void load()}
        >
          {busy ? "Computing…" : "Load Chart"}
        </button>
      </div>

      {err && <div className="card card-error">{err}</div>}

      {data && (
        <>
          <p className="muted">
            Plotted {data.total_items ?? data.items.length} with embeddings (of{" "}
            {data.total_in_history ?? "—"} in history), {data.n_clusters}{" "}
            clusters.
          </p>
          <div className="card">
            <ClusterChart
              items={data.items}
              clusterSummaries={data.cluster_summaries}
            />
          </div>

          <h2 className="section-title">Clusters</h2>
          <div className="cluster-expand-list">
            {Object.entries(data.cluster_summaries).map(([cid, s]) => {
              const items = itemsByCluster.get(cid) ?? [];
              const open = openCluster === cid;
              return (
                <div key={cid} className="cluster-expand card">
                  <button
                    type="button"
                    className="cluster-expand-head"
                    onClick={() =>
                      setOpenCluster(open ? null : cid)
                    }
                  >
                    <strong>{s.name ?? `Cluster ${cid}`}</strong>
                    <span className="muted">
                      {s.size} titles · {s.top_genres?.join(", ") ?? ""}
                    </span>
                    <span className="cluster-chevron">{open ? "▼" : "▶"}</span>
                  </button>
                  {open && items.length > 0 && (
                    <div className="cluster-items-grid">
                      {items.map((it) => {
                        const title = it.title ?? String(it.id ?? "");
                        const id = Number(it.id);
                        const mt = String(it.media_type ?? "movie");
                        const mtEnc = encodeURIComponent(
                          mt === "tv" ? "tv" : "movie",
                        );
                        const src =
                          posterUrl(it.poster_path ?? null, "w185") ??
                          placeholderPoster(title);
                        return (
                          <div
                            key={`${it.id}-${it.title}`}
                            className="cluster-item"
                          >
                            <img src={src} alt="" loading="lazy" />
                            <span className="cluster-item-title" title={title}>
                              {title}
                            </span>
                            {id > 0 && (
                              <Link
                                className="btn btn-ghost btn-small cluster-item-link"
                                to={`/similar?id=${id}&type=${mtEnc}`}
                              >
                                Similar Titles
                              </Link>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
