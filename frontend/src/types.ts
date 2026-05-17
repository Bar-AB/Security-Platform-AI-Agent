export type QueryType = 'data' | 'doc' | 'mixed'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  queryType?: QueryType
}
