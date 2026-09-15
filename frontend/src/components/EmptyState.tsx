import { EXAMPLE_GROUPS } from '../examples'
import { CategoryIcon } from './CategoryIcon'
import { ShieldIcon } from './Icons'

interface Props {
  onSelect: (question: string) => void
  disabled: boolean
}

export function EmptyState({ onSelect, disabled }: Props) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-8 py-8 text-center">
      <div className="flex flex-col items-center gap-4">
        <div className="hero-glow flex h-16 w-16 items-center justify-center rounded-2xl border border-accent/40 bg-accent/10 text-accent">
          <ShieldIcon size={34} />
        </div>
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-ink">
            How can I help secure your platform?
          </h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted">
            Investigate security issues, application risk, and CI/CD pipeline findings, or ask how
            to use the platform.
          </p>
        </div>
      </div>

      <section aria-label="Starter questions" className="grid w-full gap-3 sm:grid-cols-2">
        {EXAMPLE_GROUPS.map(group => (
          <button
            key={group.category}
            type="button"
            onClick={() => onSelect(group.questions[0])}
            disabled={disabled}
            className="focus-ring group flex items-start gap-3 rounded-xl border border-line bg-surface/80 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-accent/50 hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
          >
            <CategoryIcon category={group.category} size={32} />
            <span className="min-w-0">
              <span className="block text-sm font-medium text-ink">{group.starterTitle}</span>
              <span className="mt-0.5 block text-[13px] leading-snug text-muted group-hover:text-ink/80">
                {group.questions[0]}
              </span>
            </span>
          </button>
        ))}
      </section>
    </div>
  )
}
