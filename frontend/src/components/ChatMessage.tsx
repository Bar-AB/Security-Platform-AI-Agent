import Markdown from 'react-markdown'
import type { Message, QueryType } from '../types'

const BADGE_LABEL: Record<QueryType, string> = {
  data: 'MCP',
  doc: 'RAG',
  mixed: 'Mixed',
}

const BADGE_COLOR: Record<QueryType, string> = {
  data: 'bg-[#388bfd] text-white',
  doc: 'bg-[#3fb950] text-white',
  mixed: 'bg-[#d29922] text-white',
}

interface Props {
  message: Message
}

function ConfidenceBadge({ score, flagged }: { score: number; flagged: boolean }) {
  const pct = Math.round(score * 100)
  const color = flagged
    ? 'bg-[#da3633]/20 text-[#f85149] border border-[#da3633]/40'
    : 'bg-[#238636]/20 text-[#3fb950] border border-[#238636]/40'
  const icon = flagged ? '⚠' : '✓'
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>
      {icon} {pct}% grounded
    </span>
  )
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user'

  if (!isUser && !message.content) return null

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[75%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-[#1f6feb] text-white rounded-br-sm'
              : `bg-[#21262d] text-[#e6edf3] rounded-bl-sm${message.isStreaming ? '' : ' prose prose-sm prose-invert max-w-none'}`
          }`}
        >
          {isUser ? (
            message.content
          ) : message.isStreaming ? (
            <span className="whitespace-pre-wrap">
              {message.content}
              <span className="streaming-cursor" />
            </span>
          ) : (
            <Markdown>{message.content}</Markdown>
          )}
        </div>

        {!message.isStreaming && message.chartImage && (
          <img
            src={`data:image/png;base64,${message.chartImage}`}
            alt="Security chart"
            className="rounded-lg border border-[#30363d] max-w-full mt-1"
          />
        )}

        {!message.isStreaming && (
          <div className="flex gap-2 items-center flex-wrap">
            {!isUser && message.queryType && BADGE_LABEL[message.queryType] && (
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium ${BADGE_COLOR[message.queryType]}`}
              >
                {BADGE_LABEL[message.queryType]}
              </span>
            )}
            {!isUser && message.confidenceScore !== undefined && (
              <ConfidenceBadge
                score={message.confidenceScore}
                flagged={message.validationFlagged ?? false}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
