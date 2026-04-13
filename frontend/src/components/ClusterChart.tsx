import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  Cell,
} from "recharts";
import type { ClusterItem, ClusterSummaries } from "../api/watcher";

const PALETTE = [
  "#8b7cf6",
  "#34d399",
  "#fbbf24",
  "#f472b6",
  "#38bdf8",
  "#a78bfa",
  "#fb923c",
  "#4ade80",
  "#f87171",
  "#22d3ee",
  "#c084fc",
  "#2dd4bf",
];

type Row = ClusterItem & { fill: string; label: string };

function buildRows(items: ClusterItem[], summaries: ClusterSummaries): Row[] {
  return items.map((it) => {
    const name = summaries[String(it.cluster)]?.name ?? `Cluster ${it.cluster}`;
    return {
      ...it,
      fill: PALETTE[Math.abs(it.cluster) % PALETTE.length],
      label: name,
    };
  });
}

export function ClusterChart({
  items,
  clusterSummaries,
}: {
  items: ClusterItem[];
  clusterSummaries: ClusterSummaries;
}) {
  if (!items.length) {
    return <p className="muted">No Points To Plot.</p>;
  }

  const data = buildRows(items, clusterSummaries);

  return (
    <div className="cluster-chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 16, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e38" />
          <XAxis type="number" dataKey="x" name="x" stroke="#9aa0a8" tick={{ fill: "#9aa0a8", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="y" stroke="#9aa0a8" tick={{ fill: "#9aa0a8", fontSize: 11 }} />
          <ZAxis range={[60, 60]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as Row;
              return (
                <div className="cluster-tooltip">
                  <strong>{p.title ?? "—"}</strong>
                  <div className="muted">{p.label}</div>
                  <div className="muted">
                    {p.media_type ?? "—"} · Watched {p.watch_count ?? "—"}
                  </div>
                </div>
              );
            }}
          />
          <Scatter name="history" data={data}>
            {data.map((e, i) => (
              <Cell key={`${e.id}-${i}`} fill={e.fill} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
