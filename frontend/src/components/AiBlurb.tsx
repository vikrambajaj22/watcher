import type { ReactNode } from "react";

function capitalize(node: ReactNode): ReactNode {
  if (typeof node === "string") return node.charAt(0).toUpperCase() + node.slice(1);
  return node;
}

export function AiBlurb({ children }: { children: ReactNode }) {
  return (
    <p className="text-sm leading-relaxed m-0 px-3 py-2 bg-white/[0.05] backdrop-blur-sm rounded-lg border-l-2 border-accent/40 text-muted shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
      {capitalize(children)}
    </p>
  );
}
