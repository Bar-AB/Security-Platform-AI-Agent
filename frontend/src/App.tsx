import { useState, useRef, useEffect } from 'react'
import { ChatMessage } from './components/ChatMessage'
import { ChatInput } from './components/ChatInput'
import { TypingIndicator } from './components/TypingIndicator'
import { Sidebar } from './components/Sidebar'
import { EmptyState } from './components/EmptyState'
import { StatusIndicator, type StreamOutcome } from './components/StatusIndicator'
import { PlusIcon, ShieldIcon } from './components/Icons'
import { streamMessage } from './api'
import type { StreamDoneEvent } from './api'
import type { Message, QueryType } from './types'

const CONNECTIVITY_ERROR_PATTERN = /^(HTTP \d+|Network error|Request timed out|Stream ended unexpectedly)$/

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null)
  const [streamOutcome, setStreamOutcome] = useState<StreamOutcome | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const threadIdRef = useRef<string>(crypto.randomUUID())

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [messages])

  const handleNewChat = () => {
    if (isLoading) return
    threadIdRef.current = crypto.randomUUID()
    setMessages([])
    setStreamingStatus(null)
  }

  const reportStreamOutcome = (isOk: boolean) => {
    setStreamOutcome(prev => ({ isOk, seq: (prev?.seq ?? 0) + 1 }))
  }

  const handleSend = async (text: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setStreamingStatus(null)

    const assistantId = crypto.randomUUID()
    setMessages(prev => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', isStreaming: true },
    ])

    const accumulatedText = { current: '' }
    const pendingFrame = { current: null as number | null }
    let statusCleared = false

    await streamMessage(
      text,
      threadIdRef.current,
      (token) => {
        if (!statusCleared) {
          statusCleared = true
          setStreamingStatus(null)
        }
        accumulatedText.current += token
        if (pendingFrame.current === null) {
          pendingFrame.current = requestAnimationFrame(() => {
            pendingFrame.current = null
            setMessages(prev =>
              prev.map(m => m.id === assistantId ? { ...m, content: accumulatedText.current } : m)
            )
          })
        }
      },
      (meta: StreamDoneEvent) => {
        if (pendingFrame.current !== null) {
          cancelAnimationFrame(pendingFrame.current)
          pendingFrame.current = null
        }
        setStreamingStatus(null)
        const content = accumulatedText.current || meta.final_response || 'No response generated.'
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? {
                  ...m,
                  content,
                  isStreaming: false,
                  queryType: meta.query_type as QueryType,
                  confidenceScore: meta.confidence_score ?? undefined,
                  validationFlagged: meta.validation_flagged,
                  chartImage: meta.chart_image ?? undefined,
                }
              : m
          )
        )
        reportStreamOutcome(true)
        setIsLoading(false)
      },
      (errMsg) => {
        if (pendingFrame.current !== null) {
          cancelAnimationFrame(pendingFrame.current)
          pendingFrame.current = null
        }
        setStreamingStatus(null)
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `Error: ${errMsg}`, isStreaming: false, isError: true }
              : m
          )
        )
        if (CONNECTIVITY_ERROR_PATTERN.test(errMsg)) reportStreamOutcome(false)
        setIsLoading(false)
      },
      (statusText: string) => {
        setStreamingStatus(statusText)
      },
    )
  }

  const isAwaitingFirstToken =
    isLoading && !messages.some(m => m.isStreaming && m.content.length > 0)

  return (
    <div className="flex h-full flex-col">
      <header className="relative z-10 flex items-center gap-3 border-b border-line bg-canvas/80 px-4 py-3 backdrop-blur sm:px-6">
        <div className="brand-mark flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-canvas">
          <ShieldIcon size={20} strokeWidth={2.2} />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold tracking-tight text-ink">Security AI Agent</h1>
          <p className="hidden truncate text-xs text-muted sm:block">
            Security operations console · LangGraph · MCP · RAG
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <StatusIndicator streamOutcome={streamOutcome} />
          <button
            type="button"
            onClick={handleNewChat}
            disabled={isLoading}
            aria-label="New chat"
            className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg border border-line text-muted transition-colors hover:border-accent/50 hover:text-accent disabled:cursor-not-allowed disabled:opacity-40 md:hidden"
          >
            <PlusIcon size={16} />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar onSelect={handleSend} onNewChat={handleNewChat} disabled={isLoading} />

        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
            <div className="mx-auto h-full max-w-3xl">
              {messages.length === 0 && <EmptyState onSelect={handleSend} disabled={isLoading} />}

              {messages.length > 0 && (
                <div role="log" aria-label="Conversation">
                  {messages.map(msg => (
                    <ChatMessage key={msg.id} message={msg} />
                  ))}

                  {isAwaitingFirstToken && (
                    <div className="mb-6 pl-11">
                      <TypingIndicator status={streamingStatus} />
                    </div>
                  )}
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          <ChatInput onSend={handleSend} disabled={isLoading} />
        </main>
      </div>
    </div>
  )
}
