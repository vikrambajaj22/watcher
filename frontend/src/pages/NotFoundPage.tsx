import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center">
      <img src="/404.png" alt="404" className="w-48 rounded-2xl shadow-xl shadow-black/40 opacity-90" />
      <div>
        <h1 className="text-2xl font-bold tracking-[-0.04em] mb-2">Page not found</h1>
        <p className="text-muted mb-4">The page you're looking for doesn't exist.</p>
        <Link
          to="/"
          className="inline-flex items-center justify-center px-4 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 hover:no-underline transition-all"
        >
          Go home
        </Link>
      </div>
    </div>
  );
}
