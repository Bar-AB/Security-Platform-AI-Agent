import { useState, type KeyboardEvent } from 'react'

interface Props {
  onSend: (message: string) => void
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('')

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
    <div className="flex items-end gap-3 p-4 border-t border-[#30363d]">
      <textarea
        className="flex-1 resize-none bg-[#161b22] border border-[#30363d] rounded-xl px-4 py-3 text-[#e6edf3] text-sm placeholder-[#484f58] focus:outline-none focus:border-[#58a6ff] transition-colors"
        rows={1}
        placeholder="Ask about security issues, connectors, or the dashboard..."
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <button
        className="flex-shrink-0 px-5 py-3 bg-[#1f6feb] hover:bg-[#388bfd] disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl transition-colors"
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send"
      >
        Send
      </button>
    </div>
  )
}
