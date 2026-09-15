import { useEffect, useState } from 'react'
import { isAbortError } from '../api'

type HealthState = 'checking' | 'online' | 'offline'

export interface StreamOutcome {
  isOk: boolean
  seq: number
}

interface Props {
  streamOutcome?: StreamOutcome | null
}

const HEALTH_POLL_INTERVAL_MS = 20_000

const HEALTH_LABEL: Record<HealthState, string> = {
  checking: 'Connecting',
  online: 'Agent online',
  offline: 'Agent offline',
}

const DOT_STYLE: Record<HealthState, string> = {
  checking: 'bg-sev-medium',
  online: 'bg-ok',
  offline: 'bg-sev-critical',
}

export function StatusIndicator({ streamOutcome }: Props) {
  const [health, setHealth] = useState<HealthState>('checking')

  useEffect(() => {
    let controller = new AbortController()

    const checkHealth = () => {
      controller.abort()
      controller = new AbortController()
      fetch('/health', { signal: controller.signal })
        .then(res => {
          if (!res.ok) console.warn(`Health check returned HTTP ${res.status}`)
          setHealth(res.ok ? 'online' : 'offline')
        })
        .catch((err: unknown) => {
          if (isAbortError(err)) return
          console.warn('Health check failed:', err)
          setHealth('offline')
        })
    }

    checkHealth()
    const interval = setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS)
    return () => {
      clearInterval(interval)
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (!streamOutcome) return
    setHealth(streamOutcome.isOk ? 'online' : 'offline')
  }, [streamOutcome])

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 rounded-full border border-line bg-surface/80 px-3 py-1 text-xs text-muted"
    >
      <span className="relative flex h-2 w-2">
        {health === 'online' && (
          <span className="pulse-ring absolute inline-flex h-full w-full rounded-full bg-ok" />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${DOT_STYLE[health]}`} />
      </span>
      <span className={health === 'offline' ? 'text-sev-critical' : 'text-ink'}>
        {HEALTH_LABEL[health]}
      </span>
    </div>
  )
}
