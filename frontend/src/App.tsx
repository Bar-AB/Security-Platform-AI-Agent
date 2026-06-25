import { useState, useRef, useEffect } from 'react'
import { ChatMessage } from './components/ChatMessage'
import { ChatInput } from './components/ChatInput'
import { TypingIndicator } from './components/TypingIndicator'
import { streamMessage } from './api'
import type { StreamDoneEvent } from './api'
import type { Message, QueryType } from './types'

const THREAD_ID = crypto.randomUUID()

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [messages])

  const handleSend = async (text: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setStreamingStatus(null)

    // Placeholder assistant message updated token-by-token
    const assistantId = crypto.randomUUID()
    setMessages(prev => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', isStreaming: true },
    ])

    // Accumulate tokens in a plain object ref to avoid stale closure issues.
    // rAF batching caps DOM updates to ~60fps instead of one-per-token.
    const acc = { current: '' }
    const raf = { current: null as number | null }
    let statusCleared = false

    await streamMessage(
      text,
      THREAD_ID,
      // onToken
      (token) => {
        if (!statusCleared) {
          statusCleared = true
          setStreamingStatus(null)
        }
        acc.current += token
        if (raf.current === null) {
          raf.current = requestAnimationFrame(() => {
            raf.current = null
            setMessages(prev =>
              prev.map(m => m.id === assistantId ? { ...m, content: acc.current } : m)
            )
          })
        }
      },
      // onDone
      (meta: StreamDoneEvent) => {
        if (raf.current !== null) {
          cancelAnimationFrame(raf.current)
          raf.current = null
        }
        setStreamingStatus(null)
        const content = acc.current || meta.final_response || 'No response generated.'
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? {
                  ...m,
                  content,
                  isStreaming: false,
                  queryType: meta.query_type as QueryType,
                  confidenceScore: meta.confidence_score,
                  validationFlagged: meta.validation_flagged,
                  chartImage: meta.chart_image ?? undefined,
                }
              : m
          )
        )
        setIsLoading(false)
      },
      // onError
      (errMsg) => {
        if (raf.current !== null) {
          cancelAnimationFrame(raf.current)
          raf.current = null
        }
        setStreamingStatus(null)
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `Error: ${errMsg}`, isStreaming: false }
              : m
          )
        )
        setIsLoading(false)
      },
      // onStatus
      (statusText: string) => {
        setStreamingStatus(statusText)
      },
    )
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      <header className="flex items-center gap-3 px-6 py-4 border-b border-[#30363d]">
        <div className="w-8 h-8 rounded-lg bg-[#1f6feb] flex items-center justify-center text-white font-bold text-sm select-none">
          S
        </div>
        <div>
          <h1 className="text-[#e6edf3] font-semibold text-sm">Security AI Agent</h1>
          <p className="text-[#484f58] text-xs">Powered by GPT-4o · LangGraph · RAG</p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <div className="w-12 h-12 rounded-2xl bg-[#1f6feb]/20 flex items-center justify-center text-2xl">
              🛡️
            </div>
            <p className="text-[#484f58] text-sm max-w-sm">
              Ask about security issues, applications, pipeline findings, or platform documentation.
            </p>
            <div className="flex gap-2 flex-wrap justify-center mt-2">
              {[
                'Show critical issues',
                'How do I set up the GitHub connector?',
                'What are the top vulnerable apps?',
              ].map(hint => (
                <button
                  key={hint}
                  onClick={() => handleSend(hint)}
                  disabled={isLoading}
                  className="text-xs px-3 py-1.5 rounded-lg border border-[#30363d] text-[#58a6ff] hover:bg-[#161b22] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {hint}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && !messages.some(m => m.isStreaming && m.content.length > 0) && (
          <div className="flex justify-start mb-4">
            <div className="bg-[#21262d] rounded-2xl rounded-bl-sm">
              {streamingStatus ? (
                <p className="px-4 py-3 text-xs text-[#8b949e] italic">{streamingStatus}</p>
              ) : (
                <TypingIndicator />
              )}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  )
}
