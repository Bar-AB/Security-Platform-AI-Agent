import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import App from './App'
import * as api from './api'

vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./api')>()),
  streamMessage: vi.fn(),
}))

const mockedStream = vi.mocked(api.streamMessage)

describe('App', () => {
  beforeEach(() => {
    mockedStream.mockReset()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200 }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders the header title', async () => {
    render(<App />)
    expect(screen.getByText('Security AI Agent')).toBeInTheDocument()
    expect(await screen.findByText('Agent online')).toBeInTheDocument()
  })

  it('shows agent online when health check succeeds', async () => {
    render(<App />)
    expect(await screen.findByText('Agent online')).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledWith('/health', expect.anything())
  })

  it('shows agent offline when health check fetch fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('connection refused')))
    render(<App />)
    expect(await screen.findByText('Agent offline')).toBeInTheDocument()
  })

  it('shows agent offline when health check returns non-ok status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))
    render(<App />)
    expect(await screen.findByText('Agent offline')).toBeInTheDocument()
  })

  it('sends the starter prompt when a starter card is clicked', async () => {
    mockedStream.mockResolvedValue(undefined)
    render(<App />)
    const starters = screen.getByRole('region', { name: /starter questions/i })
    fireEvent.click(within(starters).getAllByRole('button')[0])
    await waitFor(() => expect(mockedStream).toHaveBeenCalledTimes(1))
    const sentText = mockedStream.mock.calls[0][0]
    expect(sentText.length).toBeGreaterThan(0)
    const conversation = screen.getByRole('log', { name: /conversation/i })
    expect(within(conversation).getByText(sentText)).toBeInTheDocument()
  })

  it('sends a sidebar example question when clicked', async () => {
    mockedStream.mockResolvedValue(undefined)
    render(<App />)
    const sidebar = screen.getByRole('complementary', { name: /example questions/i })
    fireEvent.click(within(sidebar).getByRole('button', { name: 'Show me all critical security issues' }))
    await waitFor(() =>
      expect(mockedStream.mock.calls[0][0]).toBe('Show me all critical security issues'),
    )
  })

  it('disables example questions while a response is loading', async () => {
    mockedStream.mockImplementation(() => new Promise(() => {}))
    render(<App />)
    const sidebar = screen.getByRole('complementary', { name: /example questions/i })
    const question = within(sidebar).getByRole('button', { name: 'Show me all critical security issues' })
    fireEvent.click(question)
    await waitFor(() => expect(question).toBeDisabled())
  })

  it('clears messages and starts a new thread when New chat is clicked', async () => {
    mockedStream.mockImplementation(async (_msg, _thread, onToken, onDone) => {
      onToken('answer text')
      onDone({ query_type: 'data', confidence_score: 1, validation_flagged: false, chart_image: null, final_response: 'answer text' })
    })
    render(<App />)
    const textbox = screen.getByRole('textbox')
    fireEvent.change(textbox, { target: { value: 'first question' } })
    fireEvent.keyDown(textbox, { key: 'Enter', shiftKey: false })
    expect(await screen.findByText('answer text')).toBeInTheDocument()

    const sidebar = screen.getByRole('complementary', { name: /example questions/i })
    fireEvent.click(within(sidebar).getByRole('button', { name: 'New chat' }))
    expect(screen.queryByText('first question')).not.toBeInTheDocument()
    expect(screen.getByRole('region', { name: /starter questions/i })).toBeInTheDocument()

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'second question' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: false })
    await waitFor(() => expect(mockedStream).toHaveBeenCalledTimes(2))
    expect(mockedStream.mock.calls[1][1]).not.toBe(mockedStream.mock.calls[0][1])
  })
  it('disables every New chat control while a stream is in flight', async () => {
    mockedStream.mockImplementation(() => new Promise(() => {}))
    render(<App />)
    const newChatButtons = screen.getAllByRole('button', { name: 'New chat' })
    expect(newChatButtons).toHaveLength(2)
    newChatButtons.forEach(button => expect(button).toBeEnabled())
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'question' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: false })
    await waitFor(() => newChatButtons.forEach(button => expect(button).toBeDisabled()))
  })

  it('renders stream errors as alerts and marks the agent offline on network failure', async () => {
    mockedStream.mockImplementation(async (_msg, _thread, _onToken, _onDone, onError) => {
      onError('Network error')
    })
    render(<App />)
    expect(await screen.findByText('Agent online')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'question' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: false })
    expect(await screen.findByRole('alert')).toHaveTextContent('Error: Network error')
    expect(await screen.findByText('Agent offline')).toBeInTheDocument()
  })

  it('keeps the agent online when the backend reports an agent error', async () => {
    mockedStream.mockImplementation(async (_msg, _thread, _onToken, _onDone, onError) => {
      onError('LLM failure')
    })
    render(<App />)
    expect(await screen.findByText('Agent online')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'question' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: false })
    expect(await screen.findByRole('alert')).toHaveTextContent('Error: LLM failure')
    expect(screen.getByText('Agent online')).toBeInTheDocument()
  })
})
