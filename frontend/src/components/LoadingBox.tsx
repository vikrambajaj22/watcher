export function LoadingBox({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-4 p-5 glass rounded-2xl mb-4" role="status" aria-live="polite">
      <div className="spinner" aria-hidden />
      <p className="text-sm m-0">{label}</p>
    </div>
  );
}
