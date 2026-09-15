import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ChatMessage } from './ChatMessage'
import type { Message } from '../types'

describe('ChatMessage', () => {
  it('renders user message content', () => {
    const msg: Message = { id: '1', role: 'user', content: 'show me issues' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('show me issues')).toBeInTheDocument()
  })

  it('renders MCP badge for data queries', () => {
    const msg: Message = { id: '2', role: 'assistant', content: 'found 2 issues', queryType: 'data' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('MCP')).toBeInTheDocument()
  })

  it('renders RAG badge for doc queries', () => {
    const msg: Message = { id: '3', role: 'assistant', content: 'here is the doc', queryType: 'doc' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('RAG')).toBeInTheDocument()
  })

  it('renders Mixed badge for mixed queries', () => {
    const msg: Message = { id: '4', role: 'assistant', content: 'combined', queryType: 'mixed' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('Mixed')).toBeInTheDocument()
  })

  it('renders Synthesis badge for synthesis queries', () => {
    const msg: Message = { id: '6', role: 'assistant', content: 'based on history', queryType: 'synthesis' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('Synthesis')).toBeInTheDocument()
  })

  it('renders error messages as an alert', () => {
    const msg: Message = { id: '7', role: 'assistant', content: 'Error: Network error', isError: true }
    render(<ChatMessage message={msg} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Error: Network error')
  })

  it('does not mark normal assistant messages as alerts', () => {
    const msg: Message = { id: '8', role: 'assistant', content: 'all good', queryType: 'data' }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('keeps the grounded confidence text', () => {
    const msg: Message = { id: '9', role: 'assistant', content: 'ok', confidenceScore: 0.87, validationFlagged: false }
    render(<ChatMessage message={msg} />)
    expect(screen.getByText(/87% grounded/)).toBeInTheDocument()
  })

  it('shows no badge for user messages', () => {
    const msg: Message = { id: '5', role: 'user', content: 'hello', queryType: 'data' }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByText('MCP')).not.toBeInTheDocument()
  })
  it('does not treat assistant content starting with Error: as an alert without isError', () => {
    const msg: Message = { id: '10', role: 'assistant', content: 'Error: log excerpt', queryType: 'data' }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByText('Error: log excerpt')).toBeInTheDocument()
  })

  it('wraps long unbroken content in the assistant card', () => {
    const url = 'https://example.com/' + 'a'.repeat(300)
    const msg: Message = { id: '11', role: 'assistant', content: url }
    const { container } = render(<ChatMessage message={msg} />)
    expect(container.querySelector('.break-words')).toHaveTextContent(url)
  })

  it('wraps long unbroken content while streaming', () => {
    const url = 'https://example.com/' + 'b'.repeat(300)
    const msg: Message = { id: '12', role: 'assistant', content: url, isStreaming: true }
    const { container } = render(<ChatMessage message={msg} />)
    expect(container.querySelector('.break-words')).toHaveTextContent(url)
  })

  it('renders a chart image when chartImage is present', () => {
    const msg: Message = { id: '13', role: 'assistant', content: 'chart', queryType: 'data', chartImage: 'QUJD' }
    render(<ChatMessage message={msg} />)
    expect(screen.getByRole('img', { name: 'Security chart' })).toHaveAttribute('src', 'data:image/png;base64,QUJD')
  })

  it('shows a fallback when the chart image fails to load', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const msg: Message = { id: '14', role: 'assistant', content: 'chart', chartImage: 'bad' }
    render(<ChatMessage message={msg} />)
    fireEvent.error(screen.getByRole('img', { name: 'Security chart' }))
    expect(screen.getByText('Chart could not be rendered')).toBeInTheDocument()
    expect(screen.queryByRole('img', { name: 'Security chart' })).not.toBeInTheDocument()
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })

  it('hides the confidence badge for blocked responses', () => {
    const msg: Message = { id: '15', role: 'assistant', content: 'blocked', queryType: 'blocked', confidenceScore: 1 }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByText(/grounded/)).not.toBeInTheDocument()
  })

  it('hides the confidence badge for chart responses', () => {
    const msg: Message = { id: '16', role: 'assistant', content: 'chart', queryType: 'chart', confidenceScore: 1 }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByText(/grounded/)).not.toBeInTheDocument()
  })

  it('renders unknown query types without crashing or a source badge', () => {
    const msg = { id: '17', role: 'assistant', content: 'mystery', queryType: 'unknown', confidenceScore: 0.9 } as unknown as Message
    render(<ChatMessage message={msg} />)
    expect(screen.getByText('mystery')).toBeInTheDocument()
    for (const label of ['MCP', 'RAG', 'Mixed', 'Synthesis']) {
      expect(screen.queryByText(label)).not.toBeInTheDocument()
    }
    expect(screen.getByText(/90% grounded/)).toBeInTheDocument()
  })

  it('renders a streaming cursor and no badges while streaming', () => {
    const msg: Message = { id: '18', role: 'assistant', content: 'partial', isStreaming: true, queryType: 'data', confidenceScore: 1 }
    const { container } = render(<ChatMessage message={msg} />)
    expect(screen.getByText('partial')).toBeInTheDocument()
    expect(container.querySelector('.streaming-cursor')).not.toBeNull()
    expect(screen.queryByText('MCP')).not.toBeInTheDocument()
  })
})
