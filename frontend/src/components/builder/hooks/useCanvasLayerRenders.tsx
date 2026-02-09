/**
 * useCanvasLayerRenders
 *
 * Builds the 12 canvas layer render functions from document state, selection,
 * tool state, and callbacks. BuilderTab composes this hook and passes the
 * return to BuilderCanvasArea. Keeps layer data-prep and render wiring out
 * of the main orchestrator.
 */

import React from 'react'
import {
  renderGrid as doRenderGrid,
  renderPreviewLine as doRenderPreviewLine,
  renderSelectionBox as doRenderSelectionBox,
  renderPolygonPreview as doRenderPolygonPreview,
  renderPathPreview as doRenderPathPreview,
  renderMeasurementMarkers as doRenderMeasurementMarkers,
  renderMeasurementLine as doRenderMeasurementLine,
  renderJoints as doRenderJoints,
  renderTrajectories as doRenderTrajectories,
  renderTargetPaths as doRenderTargetPaths,
  renderDrawnObjects as doRenderDrawnObjects,
  renderLinks as doRenderLinks
} from '../rendering'
import type { PylinkDocument } from '../types'
import type { ToolContext } from '../toolHandlers/types'
import type { HandleMergePolygonClickParams } from '../toolHandlers/mergeToolHandler'
import type { CanvasLayerRender } from '../rendering'
import type { TrajectoryStyle, ColorCycleType } from '../rendering/types'
import type { DrawnObjectsState } from '../../BuilderTools'

// Minimal state shapes used by the layer renders (structural typing)
interface MoveGroupState {
  isActive: boolean
  isDragging: boolean
  joints: string[]
  drawnObjectIds: string[]
}

interface GroupSelectionState {
  isSelecting: boolean
  startPoint: [number, number] | null
  currentPoint: [number, number] | null
}

interface PolygonDrawState {
  isDrawing: boolean
  points: [number, number][]
}

interface PathDrawState {
  isDrawing: boolean
  points: [number, number][]
}

interface MeasureState {
  isMeasuring: boolean
  startPoint: [number, number] | null
}

interface DragState {
  draggedJoint: string | null
  mergeTarget: string | null
}

interface MeasurementMarker {
  id: number
  point: [number, number]
  timestamp: number
}

interface TargetPath {
  id: string
  name: string
  points: [number, number][]
  color: string
}

interface TrajectoryData {
  trajectories: Record<string, [number, number][]>
  jointTypes?: Record<string, string>
  nSteps?: number
}

export interface UseCanvasLayerRendersParams {
  pylinkDoc: PylinkDocument
  getJointPosition: (jointName: string) => [number, number] | null
  getDefaultColor: (index: number) => string
  getHighlightStyle: (objectType: 'joint' | 'link' | 'polygon', highlightType: 'none' | 'selected' | 'hovered' | 'move_group' | 'merge', baseColor: string, baseStrokeWidth: number) => { stroke: string; strokeWidth: number; filter?: string }
  unitsToPixels: (units: number) => number
  pixelsToUnits: (pixels: number) => number
  toolMode: string
  selectedJoints: string[]
  selectedLinks: string[]
  hoveredJoint: string | null
  hoveredLink: string | null
  hoveredPolygonId: string | null
  moveGroupState: MoveGroupState
  dragState: DragState
  groupSelectionState: GroupSelectionState
  polygonDrawState: PolygonDrawState
  pathDrawState: PathDrawState
  measureState: MeasureState
  measurementMarkers: MeasurementMarker[]
  previewLine: { start: [number, number]; end: [number, number] } | null
  trajectoryData: TrajectoryData | null
  stretchingLinks: string[]
  showTrajectory: boolean
  canvasDimensions: { width: number; height: number }
  darkMode: boolean
  jointSize: number
  linkThickness: number
  trajectoryDotSize: number
  trajectoryDotOutline: boolean
  trajectoryDotOpacity: number
  trajectoryStyle: string
  trajectoryColorCycle: string
  showJointLabels: boolean
  showLinkLabels: boolean
  jointMergeRadius: number
  mergeThreshold: number
  targetPaths: TargetPath[]
  selectedPathId: string | null
  drawnObjects: DrawnObjectsState
  jointColors: { static: string; crank: string; pivot: string; moveGroup: string; mergeHighlight: string }
  getCyclicColor: (stepIndex: number, totalSteps: number, cycleType?: string) => string
  setHoveredJoint: (id: string | null) => void
  setHoveredLink: (id: string | null) => void
  setHoveredPolygonId: (id: string | null) => void
  setDrawnObjects: React.Dispatch<React.SetStateAction<DrawnObjectsState>>
  setSelectedPathId: (id: string | null) => void
  openJointEditModal: (jointName: string) => void
  openLinkEditModal: (linkName: string) => void
  handleMergeLinkClick: (context: ToolContext, linkName: string, p0: [number, number], p1: [number, number], color: string) => void
  handleMergePolygonClick: (context: ToolContext, params: HandleMergePolygonClickParams) => void | boolean
  toolContext: ToolContext
  transformPolygonPoints: (points: [number, number][], origStart: [number, number], origEnd: [number, number], currentStart: [number, number], currentEnd: [number, number]) => [number, number][]
}

export interface UseCanvasLayerRendersReturn {
  renderGrid: CanvasLayerRender
  renderDrawnObjects: CanvasLayerRender
  renderLinks: CanvasLayerRender
  renderPreviewLine: CanvasLayerRender
  renderPolygonPreview: CanvasLayerRender
  renderTargetPaths: CanvasLayerRender
  renderPathPreview: CanvasLayerRender
  renderTrajectories: CanvasLayerRender
  renderJoints: CanvasLayerRender
  renderSelectionBox: CanvasLayerRender
  renderMeasurementMarkers: CanvasLayerRender
  renderMeasurementLine: CanvasLayerRender
}

export function useCanvasLayerRenders(params: UseCanvasLayerRendersParams): UseCanvasLayerRendersReturn {
  const {
    pylinkDoc,
    getJointPosition,
    getDefaultColor,
    getHighlightStyle,
    unitsToPixels,
    pixelsToUnits,
    toolMode,
    selectedJoints,
    selectedLinks,
    hoveredJoint,
    hoveredLink,
    hoveredPolygonId,
    moveGroupState,
    dragState,
    groupSelectionState,
    polygonDrawState,
    pathDrawState,
    measureState,
    measurementMarkers,
    previewLine,
    trajectoryData,
    stretchingLinks,
    showTrajectory,
    canvasDimensions,
    darkMode,
    jointSize,
    linkThickness,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    trajectoryStyle,
    trajectoryColorCycle,
    showJointLabels,
    showLinkLabels,
    jointMergeRadius,
    mergeThreshold,
    targetPaths,
    selectedPathId,
    drawnObjects,
    jointColors,
    getCyclicColor,
    setHoveredJoint,
    setHoveredLink,
    setHoveredPolygonId,
    setDrawnObjects,
    setSelectedPathId,
    openJointEditModal,
    openLinkEditModal,
    handleMergeLinkClick,
    handleMergePolygonClick,
    toolContext,
    transformPolygonPoints
  } = params

  const renderGrid = (): React.ReactNode =>
    doRenderGrid({ canvasDimensions, darkMode, unitsToPixels, pixelsToUnits })

  const renderPreviewLine = (): React.ReactNode =>
    doRenderPreviewLine({ previewLine, unitsToPixels })

  const renderSelectionBox = (): React.ReactNode =>
    doRenderSelectionBox({
      startPoint: groupSelectionState.startPoint,
      currentPoint: groupSelectionState.currentPoint,
      isSelecting: groupSelectionState.isSelecting,
      unitsToPixels
    })

  const renderPolygonPreview = (): React.ReactNode =>
    doRenderPolygonPreview({
      points: polygonDrawState.points,
      isDrawing: polygonDrawState.isDrawing,
      mergeThreshold,
      unitsToPixels
    })

  const renderPathPreview = (): React.ReactNode =>
    doRenderPathPreview({
      points: pathDrawState.points,
      isDrawing: pathDrawState.isDrawing,
      jointMergeRadius,
      unitsToPixels
    })

  const renderMeasurementMarkers = (): React.ReactNode =>
    doRenderMeasurementMarkers({
      markers: measurementMarkers.map(m => ({ id: String(m.id), point: m.point, timestamp: m.timestamp })),
      unitsToPixels
    })

  const renderMeasurementLine = (): React.ReactNode =>
    doRenderMeasurementLine({
      startPoint: measureState.startPoint,
      isMeasuring: measureState.isMeasuring,
      unitsToPixels
    })

  const renderJoints = (): React.ReactNode => {
    const jointData = pylinkDoc.pylinkage.joints
      .map((joint, index) => {
        const pos = getJointPosition(joint.name)
        if (!pos) return null
        const meta = pylinkDoc.meta.joints[joint.name]
        const color = meta?.color || getDefaultColor(index)
        return {
          name: joint.name,
          type: joint.type as 'Static' | 'Crank' | 'Revolute',
          position: pos,
          color,
          isSelected: selectedJoints.includes(joint.name),
          isInMoveGroup: moveGroupState.isActive && moveGroupState.joints.includes(joint.name),
          isHovered: hoveredJoint === joint.name,
          isDragging: dragState.draggedJoint === joint.name,
          isMergeTarget: dragState.mergeTarget === joint.name
        }
      })
      .filter((j): j is NonNullable<typeof j> => j != null)
    return doRenderJoints({
      joints: jointData,
      jointSize,
      jointColors,
      darkMode,
      showJointLabels,
      moveGroupIsActive: moveGroupState.isActive,
      toolMode,
      getHighlightStyle,
      unitsToPixels,
      onJointHover: setHoveredJoint,
      onJointDoubleClick: openJointEditModal
    })
  }

  const renderTrajectories = (): React.ReactNode => {
    if (!trajectoryData || !showTrajectory) return null
    const trajectories = Object.entries(trajectoryData.trajectories)
      .map(([jointName, positions]): { jointName: string; positions: [number, number][]; jointType: 'Static' | 'Crank' | 'Revolute'; hasMovement: boolean; showPath: boolean } | null => {
        if (!positions || !Array.isArray(positions) || positions.length === 0) return null
        const jointType = trajectoryData.jointTypes?.[jointName]
        if (jointType !== 'Revolute' && jointType !== 'Crank') return null
        const jointMeta = pylinkDoc.meta.joints[jointName]
        if (jointMeta && jointMeta.show_path === false) return null
        const firstPos = positions[0]
        if (!firstPos || typeof firstPos[0] !== 'number' || typeof firstPos[1] !== 'number') return null
        const validPositions = positions.filter((pos): pos is [number, number] =>
          !!pos && typeof pos[0] === 'number' && typeof pos[1] === 'number' && isFinite(pos[0]) && isFinite(pos[1])
        )
        if (validPositions.length === 0) return null
        const hasMovement = validPositions.length > 1 && validPositions.some((pos, i) =>
          i > 0 && (Math.abs(pos[0] - firstPos[0]) > 0.001 || Math.abs(pos[1] - firstPos[1]) > 0.001)
        )
        return {
          jointName,
          positions: validPositions,
          jointType: jointType as 'Static' | 'Crank' | 'Revolute',
          hasMovement,
          showPath: jointMeta?.show_path !== false
        }
      })
      .filter((t): t is NonNullable<typeof t> => t != null)
    return (
      <>
        {doRenderTrajectories({
          trajectories,
          trajectoryDotSize,
          trajectoryDotOutline,
          trajectoryDotOpacity,
          trajectoryStyle: trajectoryStyle as TrajectoryStyle,
          trajectoryColorCycle: trajectoryColorCycle as ColorCycleType,
          jointColors,
          unitsToPixels,
          getCyclicColor: getCyclicColor as (stepIndex: number, totalSteps: number, cycleType?: ColorCycleType) => string
        })}
      </>
    )
  }

  const renderLinks = (): React.ReactNode => {
    const linkEntries = Object.entries(pylinkDoc.meta.links).filter(
      ([, meta]) => meta.connects.length >= 2
    ) as [string, { connects: [string, string]; color?: string; isGround?: boolean }][]

    const links: Array<{
      name: string
      connects: [string, string]
      position0: [number, number]
      position1: [number, number]
      color: string
      isGround: boolean
      isSelected: boolean
      isInMoveGroup: boolean
      isHovered: boolean
      isStretching: boolean
    }> = []

    for (let index = 0; index < linkEntries.length; index++) {
      const [linkName, linkMeta] = linkEntries[index]
      const pos0 = getJointPosition(linkMeta.connects[0])
      const pos1 = getJointPosition(linkMeta.connects[1])
      if (!pos0 || !pos1) continue

      const isGround = linkMeta.isGround ?? false
      const baseLinkColor = stretchingLinks.includes(linkName)
        ? '#ff0000'
        : isGround && !linkMeta.color
          ? '#7f7f7f'
          : (linkMeta.color || getDefaultColor(index))

      links.push({
        name: linkName,
        connects: linkMeta.connects,
        position0: pos0,
        position1: pos1,
        color: baseLinkColor,
        isGround,
        isSelected: selectedLinks.includes(linkName),
        isInMoveGroup: moveGroupState.isActive &&
          linkMeta.connects.every(j => moveGroupState.joints.includes(j)),
        isHovered: hoveredLink === linkName,
        isStretching: stretchingLinks.includes(linkName)
      })
    }

    return (
      <>
        {doRenderLinks({
          links,
          linkThickness,
          darkMode,
          showLinkLabels,
          moveGroupIsActive: moveGroupState.isActive,
          moveGroupIsDragging: moveGroupState.isDragging,
          toolMode,
          getHighlightStyle,
          unitsToPixels,
          onLinkHover: setHoveredLink,
          onLinkDoubleClick: openLinkEditModal,
          mergeMode: toolMode === 'merge',
          onMergeLinkClick: (linkName: string) => {
            const meta = pylinkDoc.meta.links[linkName]
            if (!meta?.connects || meta.connects.length < 2) return
            const p0 = getJointPosition(meta.connects[0])
            const p1 = getJointPosition(meta.connects[1])
            if (!p0 || !p1) return
            const i = Object.keys(pylinkDoc.meta.links).indexOf(linkName)
            const color = meta.color || getDefaultColor(i)
            handleMergeLinkClick(toolContext, linkName, p0, p1, color)
          },
          hitAreaStrokeWidthPx: Math.max(linkThickness * 3, 12)
        })}
      </>
    )
  }

  const renderDrawnObjects = (): React.ReactNode => {
    if (drawnObjects.objects.length === 0) return null

    const objectsWithDisplayPoints = drawnObjects.objects
      .filter((obj): obj is typeof obj & { type: 'polygon' } => obj.type === 'polygon' && obj.points.length >= 3)
      .map(obj => {
        let displayPoints = obj.points
        if (obj.mergedLinkName && obj.mergedLinkOriginalStart && obj.mergedLinkOriginalEnd) {
          const linkMeta = pylinkDoc.meta.links[obj.mergedLinkName]
          if (linkMeta) {
            const currentStart = getJointPosition(linkMeta.connects[0])
            const currentEnd = getJointPosition(linkMeta.connects[1])
            if (currentStart && currentEnd) {
              displayPoints = transformPolygonPoints(
                obj.points,
                obj.mergedLinkOriginalStart,
                obj.mergedLinkOriginalEnd,
                currentStart,
                currentEnd
              )
            }
          }
        }
        return {
          id: obj.id,
          type: 'polygon' as const,
          name: obj.name,
          points: displayPoints,
          fillColor: obj.fillColor,
          strokeColor: obj.strokeColor,
          strokeWidth: obj.strokeWidth,
          fillOpacity: obj.fillOpacity,
          mergedLinkName: obj.mergedLinkName
        }
      })

    return (
      <>
        {doRenderDrawnObjects({
          objects: objectsWithDisplayPoints,
          selectedIds: drawnObjects.selectedIds,
          moveGroupIsActive: moveGroupState.isActive,
          moveGroupDrObjectIds: moveGroupState.drawnObjectIds,
          toolMode,
          getHighlightStyle,
          unitsToPixels,
          onObjectClick: (id, _isSelected) => {
            setDrawnObjects(prev => ({
              ...prev,
              selectedIds: prev.selectedIds.includes(id)
                ? prev.selectedIds.filter(rid => rid !== id)
                : [...prev.selectedIds, id]
            }))
          },
          mergeMode: toolMode === 'merge',
          hoveredPolygonId,
          onMergePolygonHover: setHoveredPolygonId,
          onMergePolygonClick: (objId, isUnmerge) => {
            const obj = drawnObjects.objects.find(o => o.id === objId)
            if (obj) {
              handleMergePolygonClick(toolContext, {
                polygonId: obj.id,
                polygonName: obj.name ?? obj.id,
                isUnmerge,
                mergedLinkName: obj.mergedLinkName ?? undefined,
                mergedLinkOriginalStart: obj.mergedLinkOriginalStart ?? undefined,
                mergedLinkOriginalEnd: obj.mergedLinkOriginalEnd ?? undefined,
                polygonPoints: obj.points
              })
            }
          }
        })}
      </>
    )
  }

  const renderTargetPaths = (): React.ReactNode => (
    <>
      {doRenderTargetPaths({
        targetPaths: targetPaths.map(p => ({ id: p.id, name: p.name, points: p.points, color: p.color })),
        selectedPathId,
        unitsToPixels,
        onPathClick: setSelectedPathId
      })}
    </>
  )

  return {
    renderGrid,
    renderDrawnObjects,
    renderLinks,
    renderPreviewLine,
    renderPolygonPreview,
    renderTargetPaths,
    renderPathPreview,
    renderTrajectories,
    renderJoints,
    renderSelectionBox,
    renderMeasurementMarkers,
    renderMeasurementLine
  }
}
