import { CircleAlert, CircleCheck, LoaderCircle } from 'lucide-react'

interface Props {
  state: 'online' | 'offline' | 'loading'
}

export function StatusPill({ state }: Props) {
  const Icon = state === 'online' ? CircleCheck : state === 'offline' ? CircleAlert : LoaderCircle
  const label = state === 'online' ? 'Runtime online' : state === 'offline' ? 'Runtime offline' : 'Connecting'

  return (
    <span className={`status-pill ${state}`} role="status" data-testid="runtime-status">
      <Icon size={14} className={state === 'loading' ? 'spin' : ''} />
      {label}
    </span>
  )
}

