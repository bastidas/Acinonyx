/**
 * useFileInput - Reusable file input trigger and handler.
 * Use for "pick a file" flows (e.g. Load .json, Insert Canvas images).
 */
import { useRef, useCallback } from 'react'

export interface UseFileInputReturn {
  ref: React.RefObject<HTMLInputElement | null>
  trigger: () => void
  accept: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}

export function useFileInput(
  accept: string,
  onFileSelected: (file: File) => void
): UseFileInputReturn {
  const ref = useRef<HTMLInputElement>(null)

  const trigger = useCallback(() => {
    ref.current?.click()
  }, [])

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        e.target.value = ''
        onFileSelected(file)
      }
    },
    [onFileSelected]
  )

  return { ref, trigger, accept, onChange }
}
