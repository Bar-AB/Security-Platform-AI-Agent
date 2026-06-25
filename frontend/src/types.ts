export type QueryType = 'data' | 'doc' | 'mixed'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  queryType?: QueryType
  confidenceScore?: number
  validationFlagged?: boolean
  chartImage?: string
}
