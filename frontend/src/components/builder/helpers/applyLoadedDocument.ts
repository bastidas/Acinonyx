/**
 * Apply a loaded LinkageDocument to builder state.
 *
 * Call this after validating format (e.g. isHypergraphFormat). The caller is
 * responsible for showStatus and any doc patching (e.g. target_joint for demos).
 */

import type { LinkageDocument } from '../types'

export interface ApplyLoadedDocumentParams {
  doc: LinkageDocument
  setLinkageDoc: (doc: LinkageDocument) => void
  setDrawnObjects: (state: { objects: unknown[]; selectedIds: string[] }) => void
  setSelectedJoints: (ids: string[]) => void
  setSelectedLinks: (ids: string[]) => void
  clearTrajectory: () => void
  triggerMechanismChange: () => void
}

/**
 * Applies a loaded document to builder state: updates document, restores
 * drawn objects, clears selections, clears trajectory, and triggers mechanism
 * re-run. Does not call showStatus; caller handles success/error messages.
 */
export function applyLoadedDocument(params: ApplyLoadedDocumentParams): void {
  const {
    doc,
    setLinkageDoc,
    setDrawnObjects,
    setSelectedJoints,
    setSelectedLinks,
    clearTrajectory,
    triggerMechanismChange
  } = params

  setLinkageDoc(doc)
  setDrawnObjects({
    objects: Array.isArray(doc.drawnObjects) ? doc.drawnObjects : [],
    selectedIds: []
  })
  setSelectedJoints([])
  setSelectedLinks([])
  clearTrajectory()
  triggerMechanismChange()
}
