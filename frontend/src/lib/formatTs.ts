/** Format API ISO timestamps for display in the UI. */
export function formatDisplayTs(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (!Number.isNaN(d.getTime())) return d.toLocaleString();
  } catch {
    /* ignore */
  }
  return iso;
}
