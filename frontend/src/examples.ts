export type ExampleCategory = 'issues' | 'apps' | 'pipelines' | 'docs'

export interface ExampleGroup {
  category: ExampleCategory
  title: string
  starterTitle: string
  questions: string[]
}

export const EXAMPLE_GROUPS: ExampleGroup[] = [
  {
    category: 'issues',
    title: 'Security Issues',
    starterTitle: 'Triage critical findings',
    questions: [
      'Show me all critical security issues',
      'How many open issues are there by severity?',
      'What is Log4Shell and are we affected?',
    ],
  },
  {
    category: 'apps',
    title: 'Applications',
    starterTitle: 'Rank application risk',
    questions: [
      'What apps have the highest risk score?',
      'Compare auth-service and payment-service',
    ],
  },
  {
    category: 'pipelines',
    title: 'Pipelines',
    starterTitle: 'Inspect CI/CD findings',
    questions: ['Show pipeline findings in payment-service'],
  },
  {
    category: 'docs',
    title: 'Docs & Charts',
    starterTitle: 'Learn and visualize',
    questions: ['How do I configure the GitHub connector?', 'Visualize the top vulnerable applications'],
  },
]
