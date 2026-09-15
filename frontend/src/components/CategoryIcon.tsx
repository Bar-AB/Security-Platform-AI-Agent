import type { ExampleCategory } from '../examples'
import { AppsIcon, BookIcon, BugIcon, PipelineIcon } from './Icons'

const CATEGORY_STYLE: Record<ExampleCategory, string> = {
  issues: 'text-sev-critical bg-sev-critical/10 ring-sev-critical/25',
  apps: 'text-sev-high bg-sev-high/10 ring-sev-high/25',
  pipelines: 'text-sev-medium bg-sev-medium/10 ring-sev-medium/25',
  docs: 'text-accent bg-accent/10 ring-accent/25',
}

const CATEGORY_ICON = {
  issues: BugIcon,
  apps: AppsIcon,
  pipelines: PipelineIcon,
  docs: BookIcon,
}

export function CategoryIcon({ category, size = 28 }: { category: ExampleCategory; size?: number }) {
  const Icon = CATEGORY_ICON[category]
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-md ring-1 ring-inset ${CATEGORY_STYLE[category]}`}
      style={{ width: size, height: size }}
    >
      <Icon size={Math.round(size * 0.55)} />
    </span>
  )
}
