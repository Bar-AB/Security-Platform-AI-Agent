import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
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

  it('shows no badge for user messages', () => {
    const msg: Message = { id: '5', role: 'user', content: 'hello', queryType: 'data' }
    render(<ChatMessage message={msg} />)
    expect(screen.queryByText('MCP')).not.toBeInTheDocument()
  })
})
