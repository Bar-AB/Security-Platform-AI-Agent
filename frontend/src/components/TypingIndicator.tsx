export function TypingIndicator({ status }: { status: string | null }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="inline-flex items-center gap-2.5 rounded-full border border-accent/30 bg-accent/10 px-3.5 py-1.5 text-xs text-accent"
    >
      <span className="spinner h-3.5 w-3.5 rounded-full border-2 border-accent/30 border-t-accent" />
      <span>{status ?? 'Agent is working'}</span>
      <span className="flex gap-0.5" aria-hidden="true">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="typing-dot h-1 w-1 rounded-full bg-accent"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
    </div>
  )
}
