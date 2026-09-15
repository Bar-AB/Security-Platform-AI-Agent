import { useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { SendIcon } from './Icons'

interface Props {
  onSend: (message: string) => void
  disabled: boolean
}

const MAX_TEXTAREA_HEIGHT_PX = 200

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useLayoutEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`
  }, [value])

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="px-4 pb-4 pt-2 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <div className="composer flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 transition-shadow">
          <textarea
            ref={textareaRef}
            aria-label="Message"
            className="max-h-[200px] flex-1 resize-none overflow-y-auto bg-transparent px-3 py-2.5 text-sm text-ink placeholder:text-muted focus:outline-none disabled:cursor-not-allowed"
            rows={1}
            placeholder="Ask about security issues, applications, pipelines, or docs..."
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <button
            type="button"
            className="focus-ring flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent text-canvas shadow-[0_0_20px_-4px_var(--color-accent)] transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-muted disabled:shadow-none"
            onClick={submit}
            disabled={disabled || !value.trim()}
            aria-label="Send"
          >
            <SendIcon size={18} />
          </button>
        </div>
        <p className="mt-2 text-center text-[11px] text-muted">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  )
}
