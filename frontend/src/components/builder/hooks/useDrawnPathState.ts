/**
 * useDrawnPathState
 *
 * Consolidates drawn objects, target paths, path drawing, and merge state for BuilderTab
 * (Phase 6.3 — group 5: drawn/path/merge). Replaces five separate useState calls with one hook.
 */

import { useState } from 'react'
import type {
  DrawnObjectsState,
  TargetPath,
  PathDrawState,
  MergePolygonState
} from '../../BuilderTools'
import {
  initialDrawnObjectsState,
  initialPathDrawState,
  initialMergePolygonState
} from '../../BuilderTools'

export interface UseDrawnPathStateReturn {
  drawnObjects: DrawnObjectsState
  setDrawnObjects: React.Dispatch<React.SetStateAction<DrawnObjectsState>>
  targetPaths: TargetPath[]
  setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
  selectedPathId: string | null
  setSelectedPathId: React.Dispatch<React.SetStateAction<string | null>>
  pathDrawState: PathDrawState
  setPathDrawState: React.Dispatch<React.SetStateAction<PathDrawState>>
  mergePolygonState: MergePolygonState
  setMergePolygonState: React.Dispatch<React.SetStateAction<MergePolygonState>>
}

export function useDrawnPathState(): UseDrawnPathStateReturn {
  const [drawnObjects, setDrawnObjects] = useState<DrawnObjectsState>(initialDrawnObjectsState)
  const [targetPaths, setTargetPaths] = useState<TargetPath[]>([])
  const [selectedPathId, setSelectedPathId] = useState<string | null>(null)
  const [pathDrawState, setPathDrawState] = useState<PathDrawState>(initialPathDrawState)
  const [mergePolygonState, setMergePolygonState] = useState<MergePolygonState>(initialMergePolygonState)

  return {
    drawnObjects,
    setDrawnObjects,
    targetPaths,
    setTargetPaths,
    selectedPathId,
    setSelectedPathId,
    pathDrawState,
    setPathDrawState,
    mergePolygonState,
    setMergePolygonState
  }
}
