import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamMessage } from './api'

function sseResponse(lines: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line))
      controller.close()
    },
  })
  return new Response(body, { status: 200 })
}

function callbacks() {
  return { onToken: vi.fn(), onDone: vi.fn(), onError: vi.fn(), onStatus: vi.fn() }
}

async function run(cb: ReturnType<typeof callbacks>) {
  await streamMessage('hi', 't-1', cb.onToken, cb.onDone, cb.onError, cb.onStatus)
}

describe('streamMessage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('reports a network error when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const cb = callbacks()
    await run(cb)
    expect(cb.onError).toHaveBeenCalledWith('Network error')
  })

  it('reports a timeout when the request is aborted', async () => {
    const abortError = Object.assign(new Error('aborted'), { name: 'AbortError' })
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abortError))
    const cb = callbacks()
    await run(cb)
    expect(cb.onError).toHaveBeenCalledWith('Request timed out')
  })

  it('reports an error when the stream ends without a done event', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      sseResponse(['data: {"type":"token","content":"partial"}\n\n']),
    ))
    const cb = callbacks()
    await run(cb)
    expect(cb.onToken).toHaveBeenCalledWith('partial')
    expect(cb.onError).toHaveBeenCalledWith('Stream ended unexpectedly')
  })

  it('completes without error when a done event arrives', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      sseResponse(['data: {"type":"done","query_type":"data","confidence_score":0.9}\n\n']),
    ))
    const cb = callbacks()
    await run(cb)
    expect(cb.onDone).toHaveBeenCalledWith(expect.objectContaining({ query_type: 'data' }))
    expect(cb.onError).not.toHaveBeenCalled()
  })

  it('warns about malformed lines and keeps reading', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      sseResponse(['data: {not json\n\n', 'data: {"type":"done","query_type":"doc"}\n\n']),
    ))
    const cb = callbacks()
    await run(cb)
    expect(warn).toHaveBeenCalled()
    expect(cb.onDone).toHaveBeenCalledWith(expect.objectContaining({ query_type: 'doc' }))
  })

  it('resolves quietly when the stream fails after a done event', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const encoder = new TextEncoder()
    let pulls = 0
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulls += 1
        if (pulls === 1) {
          controller.enqueue(encoder.encode('data: {"type":"done","query_type":"data"}\n\n'))
          return
        }
        controller.error(new TypeError('connection reset'))
      },
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status: 200 })))
    const cb = callbacks()
    await expect(run(cb)).resolves.toBeUndefined()
    expect(cb.onDone).toHaveBeenCalled()
    expect(cb.onError).not.toHaveBeenCalled()
    expect(warn).toHaveBeenCalled()
  })

  it('passes through an unknown confidence score as null', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      sseResponse(['data: {"type":"done","query_type":"data","confidence_score":null}\n\n']),
    ))
    const cb = callbacks()
    await run(cb)
    expect(cb.onDone).toHaveBeenCalledWith(expect.objectContaining({ confidence_score: null }))
  })
})
