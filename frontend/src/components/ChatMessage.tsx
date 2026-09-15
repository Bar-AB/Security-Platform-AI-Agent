import { useState, type ComponentType } from 'react'
import Markdown, { type Components } from 'react-markdown'
import type { Message, QueryType } from '../types'
import {
  AlertIcon,
  BookIcon,
  CheckIcon,
  DatabaseIcon,
  LayersIcon,
  ShieldIcon,
  SparkIcon,
  UserIcon,
} from './Icons'

type SourceQueryType = 'data' | 'doc' | 'mixed' | 'synthesis'

const UNVALIDATED_QUERY_TYPES: ReadonlySet<QueryType> = new Set<QueryType>(['blocked', 'chart'])

const BADGE_LABEL: Record<SourceQueryType, string> = {
  data: 'MCP',
  doc: 'RAG',
  mixed: 'Mixed',
  synthesis: 'Synthesis',
}

const BADGE_COLOR: Record<SourceQueryType, string> = {
  data: 'text-accent bg-accent/10 ring-accent/30',
  doc: 'text-ok bg-ok/10 ring-ok/30',
  mixed: 'text-sev-medium bg-sev-medium/10 ring-sev-medium/30',
  synthesis: 'text-violet bg-violet/10 ring-violet/30',
}

const BADGE_ICON: Record<SourceQueryType, ComponentType<{ size?: number }>> = {
  data: DatabaseIcon,
  doc: BookIcon,
  mixed: LayersIcon,
  synthesis: SparkIcon,
}

const MARKDOWN_COMPONENTS: Components = {
  table: ({ node: _node, ...props }) => (
    <div className="table-scroll">
      <table {...props} />
    </div>
  ),
}

interface Props {
  message: Message
}

function ConfidenceBadge({ score, flagged }: { score: number; flagged: boolean }) {
  const pct = Math.round(score * 100)
  const color = flagged
    ? 'text-sev-critical bg-sev-critical/10 ring-sev-critical/30'
    : 'text-ok bg-ok/10 ring-ok/30'
  const Icon = flagged ? AlertIcon : CheckIcon
  return (
    <span className={`badge ring-1 ring-inset ${color}`}>
      <Icon size={12} />
      {pct}% grounded
    </span>
  )
}

function isSourceQueryType(queryType: QueryType | undefined): queryType is SourceQueryType {
  return queryType !== undefined && Object.prototype.hasOwnProperty.call(BADGE_LABEL, queryType)
}

function SourceBadge({ queryType }: { queryType: SourceQueryType }) {
  const Icon = BADGE_ICON[queryType]
  return (
    <span className={`badge ring-1 ring-inset ${BADGE_COLOR[queryType]}`}>
      <Icon size={12} />
      {BADGE_LABEL[queryType]}
    </span>
  )
}

function ChartFigure({ image }: { image: string }) {
  const [hasFailed, setHasFailed] = useState(false)

  if (hasFailed) {
    return (
      <div className="rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted">
        Chart could not be rendered
      </div>
    )
  }

  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-surface p-2">
      <img
        src={`data:image/png;base64,${image}`}
        alt="Security chart"
        className="max-w-full rounded-lg bg-white"
        onError={() => {
          console.warn('Chart image failed to render')
          setHasFailed(true)
        }}
      />
    </figure>
  )
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="message-enter mb-6 flex justify-end gap-3">
      <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md border border-accent/30 bg-accent/15 px-4 py-2.5 text-sm leading-relaxed text-ink sm:max-w-[75%]">
        {content}
      </div>
      <span className="mt-0.5 hidden h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line bg-surface-2 text-muted sm:flex">
        <UserIcon size={16} />
      </span>
    </div>
  )
}

export function ChatMessage({ message }: Props) {
  if (message.role === 'user') return <UserMessage content={message.content} />
  if (!message.content) return null

  const isError = !message.isStreaming && message.isError === true
  const hasSourceBadge = isSourceQueryType(message.queryType)
  const hasConfidenceBadge =
    message.confidenceScore !== undefined &&
    !(message.queryType !== undefined && UNVALIDATED_QUERY_TYPES.has(message.queryType))
  const hasBadges = !message.isStreaming && (hasSourceBadge || hasConfidenceBadge)

  return (
    <div className="message-enter mb-6 flex justify-start gap-3">
      <span
        className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${
          isError
            ? 'border-sev-critical/40 bg-sev-critical/10 text-sev-critical'
            : 'avatar-glow border-accent/40 bg-accent/10 text-accent'
        }`}
      >
        {isError ? <AlertIcon size={16} /> : <ShieldIcon size={16} />}
      </span>

      <div className="flex min-w-0 flex-1 flex-col gap-2">
        {isError ? (
          <div
            role="alert"
            className="break-words rounded-xl border border-sev-critical/40 bg-sev-critical/10 px-4 py-3 text-sm text-sev-critical"
          >
            {message.content}
          </div>
        ) : (
          <div className="break-words rounded-xl border border-line bg-surface/90 px-4 py-3 text-sm leading-relaxed text-ink shadow-lg shadow-black/20">
            {message.isStreaming ? (
              <span className="whitespace-pre-wrap break-words">
                {message.content}
                <span className="streaming-cursor" />
              </span>
            ) : (
              <div className="markdown prose prose-sm prose-invert max-w-none">
                <Markdown components={MARKDOWN_COMPONENTS}>{message.content}</Markdown>
              </div>
            )}
          </div>
        )}

        {!message.isStreaming && message.chartImage && (
          <ChartFigure image={message.chartImage} />
        )}

        {hasBadges && (
          <div className="flex flex-wrap items-center gap-2">
            {isSourceQueryType(message.queryType) && <SourceBadge queryType={message.queryType} />}
            {hasConfidenceBadge && message.confidenceScore !== undefined && (
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
