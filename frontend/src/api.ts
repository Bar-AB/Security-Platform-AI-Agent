export interface ChatApiResponse {
  response: string
  query_type: string
}

export async function sendMessage(message: string, threadId: string): Promise<ChatApiResponse> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<ChatApiResponse>
}
