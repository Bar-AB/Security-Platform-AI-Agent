export interface ChatApiResponse {
  response: string
  query_type: string
  confidence_score: number
  validation_flagged: boolean
  chart_image: string | null
}

export async function sendMessage(message: string, threadId: string): Promise<ChatApiResponse> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 30_000)
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json() as Promise<ChatApiResponse>
  } finally {
    clearTimeout(timer)
  }
}

export interface StreamDoneEvent {
  query_type: string
  confidence_score: number
  validation_flagged: boolean
  chart_image: string | null
  final_response?: string
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
  const timer = setTimeout(() => controller.abort(), 60_000)
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
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const raw = line.slice(6).trim()
        if (!raw) continue
        try {
          const evt = JSON.parse(raw) as { type: string; content?: string; text?: string } & Partial<StreamDoneEvent>
          if (evt.type === 'status' && evt.text) {
            onStatus?.(evt.text)
          } else if (evt.type === 'token' && evt.content) {
            onToken(evt.content)
          } else if (evt.type === 'done') {
            onDone({
              query_type: evt.query_type ?? 'unknown',
              confidence_score: evt.confidence_score ?? 1.0,
              validation_flagged: evt.validation_flagged ?? false,
              chart_image: evt.chart_image ?? null,
              final_response: evt.final_response,
            })
          } else if (evt.type === 'error') {
            onError(evt.content ?? 'Unknown stream error')
          }
        } catch {
          // ignore malformed SSE line
        }
      }
    }
  } finally {
    clearTimeout(timer)
  }
}
