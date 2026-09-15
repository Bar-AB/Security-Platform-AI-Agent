export type QueryType = 'data' | 'doc' | 'mixed' | 'synthesis' | 'blocked' | 'chart'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  isError?: boolean
  queryType?: QueryType
  confidenceScore?: number
  validationFlagged?: boolean
  chartImage?: string
}
