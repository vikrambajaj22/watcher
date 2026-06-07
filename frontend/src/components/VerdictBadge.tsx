type Props = { willLike: boolean; score?: number };

export function VerdictBadge({ willLike, score }: Props) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`px-2.5 py-1 rounded-full text-sm font-semibold whitespace-nowrap ${
          willLike
            ? "bg-emerald-400/15 text-emerald-300 border border-emerald-400/20"
            : "bg-yellow-400/10 text-yellow-300 border border-yellow-400/20"
        }`}
      >
        {willLike ? "Likely Yes" : "Probably Not"}
      </span>
      {score !== undefined && (
        <span className="text-sm text-muted">{(score * 100).toFixed(0)}% match</span>
      )}
    </div>
  );
}
