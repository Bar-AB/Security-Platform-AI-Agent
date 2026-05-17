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

export function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[75%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-[#1f6feb] text-white rounded-br-sm'
              : 'bg-[#21262d] text-[#e6edf3] rounded-bl-sm prose prose-sm prose-invert max-w-none'
          }`}
        >
          {isUser ? message.content : <Markdown>{message.content}</Markdown>}
        </div>
        {!isUser && message.queryType && BADGE_LABEL[message.queryType] && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${BADGE_COLOR[message.queryType]}`}
          >
            {BADGE_LABEL[message.queryType]}
          </span>
        )}
      </div>
    </div>
  )
}
