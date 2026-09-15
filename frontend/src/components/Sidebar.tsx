import { EXAMPLE_GROUPS } from '../examples'
import { CategoryIcon } from './CategoryIcon'
import { PlusIcon } from './Icons'

interface Props {
  onSelect: (question: string) => void
  onNewChat: () => void
  disabled: boolean
}

export function Sidebar({ onSelect, onNewChat, disabled }: Props) {
  return (
    <aside
      aria-label="Example questions"
      className="hidden w-72 shrink-0 flex-col border-r border-line bg-surface/60 backdrop-blur md:flex"
    >
      <div className="p-4">
        <button
          type="button"
          onClick={onNewChat}
          disabled={disabled}
          className="focus-ring flex w-full items-center justify-center gap-2 rounded-lg border border-accent/40 bg-accent/10 px-3 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <PlusIcon size={16} />
          New chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 pb-6">
        {EXAMPLE_GROUPS.map(group => (
          <section key={group.category} className="mb-5">
            <h2 className="mb-2 flex items-center gap-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-muted">
              <CategoryIcon category={group.category} size={20} />
              {group.title}
            </h2>
            <ul className="space-y-0.5">
              {group.questions.map(question => (
                <li key={question}>
                  <button
                    type="button"
                    onClick={() => onSelect(question)}
                    disabled={disabled}
                    className="focus-ring w-full rounded-md px-2 py-1.5 text-left text-[13px] leading-snug text-ink/80 transition-colors hover:bg-surface-2 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {question}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </nav>

      <p className="border-t border-line px-4 py-3 text-[11px] text-muted">
        Mock platform data · LangGraph · MCP · RAG
      </p>
    </aside>
  )
}
