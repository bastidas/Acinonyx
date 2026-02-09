/**
 * useDocumentState
 *
 * Consolidates document (linkage) state for BuilderTab (Phase 6 — group 4).
 * Single source of truth for the hypergraph LinkageDocument.
 */

import { useState } from 'react'
import type { LinkageDocument } from '../types'
import { createEmptyLinkageDocument } from '../conversions'

export interface UseDocumentStateReturn {
  linkageDoc: LinkageDocument
  setLinkageDoc: React.Dispatch<React.SetStateAction<LinkageDocument>>
}

export function useDocumentState(
  initialDoc?: LinkageDocument
): UseDocumentStateReturn {
  const [linkageDoc, setLinkageDoc] = useState<LinkageDocument>(
    () => initialDoc ?? createEmptyLinkageDocument()
  )

  return {
    linkageDoc,
    setLinkageDoc
  }
}
