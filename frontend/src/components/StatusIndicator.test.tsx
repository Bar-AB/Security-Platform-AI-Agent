import { act, render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { StatusIndicator } from './StatusIndicator'

describe('StatusIndicator', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('polls health and flips to offline when a later check fails', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockRejectedValueOnce(new Error('connection refused'))
    vi.stubGlobal('fetch', fetchMock)
    render(<StatusIndicator />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByText('Agent online')).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Agent offline')).toBeInTheDocument()
  })

  it('flips back online when a later poll succeeds', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValueOnce({ ok: true, status: 200 })
    vi.stubGlobal('fetch', fetchMock)
    render(<StatusIndicator />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByText('Agent offline')).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })
    expect(screen.getByText('Agent online')).toBeInTheDocument()
  })

  it('stops polling and aborts the in-flight check on unmount', async () => {
    const signals: AbortSignal[] = []
    const fetchMock = vi.fn((_url: string, init: RequestInit) => {
      signals.push(init.signal as AbortSignal)
      return new Promise(() => {})
    })
    vi.stubGlobal('fetch', fetchMock)
    const { unmount } = render(<StatusIndicator />)
    unmount()
    expect(signals[0].aborted).toBe(true)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('shows offline when a stream failure is reported', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200 }))
    const { rerender } = render(<StatusIndicator />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByText('Agent online')).toBeInTheDocument()

    rerender(<StatusIndicator streamOutcome={{ isOk: false, seq: 1 }} />)
    expect(screen.getByText('Agent offline')).toBeInTheDocument()

    rerender(<StatusIndicator streamOutcome={{ isOk: true, seq: 2 }} />)
    expect(screen.getByText('Agent online')).toBeInTheDocument()
  })
})
