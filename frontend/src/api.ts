export interface StreamDoneEvent {
  query_type: string
  confidence_score: number | null
  validation_flagged: boolean
  chart_image: string | null
  final_response?: string
}

type StreamEvent = { type: string; content?: string; text?: string } & Partial<StreamDoneEvent>

const IDLE_TIMEOUT_MS = 60_000
const SSE_DATA_PREFIX = 'data: '

function parseSseLine(line: string): StreamEvent | null {
  if (!line.startsWith(SSE_DATA_PREFIX)) return null
  const raw = line.slice(SSE_DATA_PREFIX.length).trim()
  if (!raw) return null
  try {
    return JSON.parse(raw) as StreamEvent
  } catch {
    console.warn('Ignoring malformed SSE line:', raw)
    return null
  }
}

function isAbortError(err: unknown): boolean {
  return typeof err === 'object' && err !== null && (err as { name?: string }).name === 'AbortError'
}

export async function streamMessage(
  message: string,
  threadId: string,
  onToken: (token: string) => void,
  onDone: (meta: StreamDoneEvent) => void,
  onError: (msg: string) => void,
  onStatus?: (text: string) => void,
): Promise<void> {
  const controller = new AbortController()
  let timer = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS)
  const restartIdleTimer = () => {
    clearTimeout(timer)
    timer = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS)
  }
  let isFinished = false

  const dispatch = (evt: StreamEvent) => {
    if (evt.type === 'status' && evt.text) {
      onStatus?.(evt.text)
    } else if (evt.type === 'token' && evt.content) {
      onToken(evt.content)
    } else if (evt.type === 'done') {
      isFinished = true
      onDone({
        query_type: evt.query_type ?? 'unknown',
        confidence_score: evt.confidence_score === undefined ? 1.0 : evt.confidence_score,
        validation_flagged: evt.validation_flagged ?? false,
        chart_image: evt.chart_image ?? null,
        final_response: evt.final_response,
      })
    } else if (evt.type === 'error') {
      isFinished = true
      onError(evt.content ?? 'Unknown stream error')
    }
  }

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal: controller.signal,
    })
    if (!res.ok || !res.body) {
      onError(`HTTP ${res.status}`)
      return
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      restartIdleTimer()
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const evt = parseSseLine(line)
        if (evt) dispatch(evt)
      }
    }
    if (!isFinished) onError('Stream ended unexpectedly')
  } catch (err) {
    if (isFinished) {
      console.warn('Stream error after completion:', err)
      return
    }
    onError(isAbortError(err) ? 'Request timed out' : 'Network error')
  } finally {
    clearTimeout(timer)
  }
}
