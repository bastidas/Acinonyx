/**
 * useToolFlowsState
 *
 * Consolidates tool-flow state for BuilderTab: link creation, preview line, drag,
 * group selection, polygon drawing, measure, measurement markers, move group
 * (Phase 6.3 — group 6). Replaces eight separate useState calls with one hook.
 */

import { useState } from 'react'
import type {
  LinkCreationState,
  DragState,
  GroupSelectionState,
  PolygonDrawState,
  MeasureState,
  MeasurementMarker,
  MoveGroupState
} from '../../BuilderTools'
import {
  initialLinkCreationState,
  initialDragState,
  initialGroupSelectionState,
  initialPolygonDrawState,
  initialMeasureState,
  initialMoveGroupState
} from '../../BuilderTools'

export type PreviewLine = { start: [number, number]; end: [number, number] } | null

export interface UseToolFlowsStateReturn {
  linkCreationState: LinkCreationState
  setLinkCreationState: React.Dispatch<React.SetStateAction<LinkCreationState>>
  previewLine: PreviewLine
  setPreviewLine: React.Dispatch<React.SetStateAction<PreviewLine>>
  dragState: DragState
  setDragState: React.Dispatch<React.SetStateAction<DragState>>
  groupSelectionState: GroupSelectionState
  setGroupSelectionState: React.Dispatch<React.SetStateAction<GroupSelectionState>>
  polygonDrawState: PolygonDrawState
  setPolygonDrawState: React.Dispatch<React.SetStateAction<PolygonDrawState>>
  measureState: MeasureState
  setMeasureState: React.Dispatch<React.SetStateAction<MeasureState>>
  measurementMarkers: MeasurementMarker[]
  setMeasurementMarkers: React.Dispatch<React.SetStateAction<MeasurementMarker[]>>
  moveGroupState: MoveGroupState
  setMoveGroupState: React.Dispatch<React.SetStateAction<MoveGroupState>>
}

export function useToolFlowsState(): UseToolFlowsStateReturn {
  const [linkCreationState, setLinkCreationState] = useState<LinkCreationState>(initialLinkCreationState)
  const [previewLine, setPreviewLine] = useState<PreviewLine>(null)
  const [dragState, setDragState] = useState<DragState>(initialDragState)
  const [groupSelectionState, setGroupSelectionState] = useState<GroupSelectionState>(initialGroupSelectionState)
  const [polygonDrawState, setPolygonDrawState] = useState<PolygonDrawState>(initialPolygonDrawState)
  const [measureState, setMeasureState] = useState<MeasureState>(initialMeasureState)
  const [measurementMarkers, setMeasurementMarkers] = useState<MeasurementMarker[]>([])
  const [moveGroupState, setMoveGroupState] = useState<MoveGroupState>(initialMoveGroupState)

  return {
    linkCreationState,
    setLinkCreationState,
    previewLine,
    setPreviewLine,
    dragState,
    setDragState,
    groupSelectionState,
    setGroupSelectionState,
    polygonDrawState,
    setPolygonDrawState,
    measureState,
    setMeasureState,
    measurementMarkers,
    setMeasurementMarkers,
    moveGroupState,
    setMoveGroupState
  }
}
