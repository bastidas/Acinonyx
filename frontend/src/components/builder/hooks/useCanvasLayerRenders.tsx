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
  renderCanvasImages as doRenderCanvasImages,
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
  renderLinks as doRenderLinks,
  renderExplorationDots as doRenderExplorationDots,
  renderExplorationTrajectories as doRenderExplorationTrajectories,
  filterTrajectoriesForRendering
} from '../rendering'
import type { PylinkDocument } from '../types'
import type { ToolContext } from '../toolHandlers/types'
import type { HandleMergePolygonClickParams } from '../toolHandlers/mergeToolHandler'
import type { CanvasLayerRender } from '../rendering'
import type { TrajectoryStyle, ColorCycleType } from '../rendering/types'
import type { DrawnObjectsState, ExploreTrajectoriesState } from '../../BuilderTools'
import type { CanvasImageData } from '../types'

/** One entry per unique position; valid if any sample at that position is valid. */
function deduplicateBySecondPosition(
  samples: Array<{ position: [number, number]; valid: boolean }>
): Array<{ position: [number, number]; valid: boolean }> {
  const key = (p: [number, number]) => `${p[0].toFixed(6)},${p[1].toFixed(6)}`
  const map = new Map<string, { position: [number, number]; valid: boolean }>()
  for (const s of samples) {
    const k = key(s.position)
    const existing = map.get(k)
    if (!existing) map.set(k, { position: s.position, valid: s.valid })
    else if (s.valid) map.set(k, { position: s.position, valid: true })
  }
  return Array.from(map.values())
}

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
  jointOutline: number
  linkThickness: number
  linkTransparency: number
  linkColorMode: 'various' | 'z-level' | 'single'
  linkColorSingle: string
  trajectoryDotSize: number
  trajectoryDotOutline: boolean
  trajectoryDotOpacity: number
  showTrajectoryStepNumbers: boolean
  trajectoryStyle: string
  trajectoryColorCycle: string
  showJointLabels: boolean
  showLinkLabels: boolean
  jointMergeRadius: number
  mergeThreshold: number
  targetPaths: TargetPath[]
  selectedPathId: string | null
  drawnObjects: DrawnObjectsState
  canvases: CanvasImageData[]
  setCanvases: React.Dispatch<React.SetStateAction<CanvasImageData[]>>
  openCanvasEdit: (id: string) => void
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
  exploreTrajectoriesState: ExploreTrajectoriesState
  exploreColormapEnabled?: boolean
  exploreColormapType?: 'rainbow' | 'twilight' | 'husl'
  exploreRadius?: number
}

export interface UseCanvasLayerRendersReturn {
  renderGrid: CanvasLayerRender
  renderCanvases: CanvasLayerRender
  renderDrawnObjects: CanvasLayerRender
  renderLinks: CanvasLayerRender
  renderPreviewLine: CanvasLayerRender
  renderPolygonPreview: CanvasLayerRender
  renderTargetPaths: CanvasLayerRender
  renderPathPreview: CanvasLayerRender
  renderExplorationTrajectories: CanvasLayerRender
  renderTrajectories: CanvasLayerRender
  renderExplorationDots: CanvasLayerRender
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
    jointOutline,
    linkThickness,
    linkTransparency,
    linkColorMode,
    linkColorSingle,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    showTrajectoryStepNumbers,
    trajectoryStyle,
    trajectoryColorCycle,
    showJointLabels,
    showLinkLabels,
    jointMergeRadius,
    mergeThreshold,
    targetPaths,
    selectedPathId,
    drawnObjects,
    canvases,
    setCanvases,
    openCanvasEdit,
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
  transformPolygonPoints,
  exploreTrajectoriesState,
  exploreColormapEnabled = false,
  exploreColormapType = 'rainbow',
  exploreRadius = 20
} = params

  const renderGrid = (): React.ReactNode =>
    doRenderGrid({ canvasDimensions, darkMode, unitsToPixels, pixelsToUnits })

  const renderCanvases = (): React.ReactNode =>
    canvases.length === 0 ? null : (
      <g data-layer="canvas-images">
        {doRenderCanvasImages({
          canvases,
          unitsToPixels,
          pixelsToUnits,
          onRequestEdit: openCanvasEdit,
          onPositionChange: (id, position) => {
            setCanvases(prev => prev.map(c => (c.id === id ? { ...c, position } : c)))
          }
        })}
      </g>
    )

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

  const renderExplorationTrajectories = (): React.ReactNode => {
    if (toolMode !== 'explore_node_trajectories' || !exploreTrajectoriesState.exploreSamples.length) return null
    const jointNamesToShow = Object.entries(pylinkDoc.meta.joints)
      .filter(([, m]) => m?.show_path === true)
      .map(([name]) => name)
    const exploreCenter =
      exploreTrajectoriesState.exploreMode === 'combinatorial'
        ? exploreTrajectoriesState.exploreSecondCenter
        : exploreTrajectoriesState.exploreCenter
    const hoveredPosition =
      exploreTrajectoriesState.exploreHoveredIndex != null && exploreTrajectoriesState.exploreSamples[exploreTrajectoriesState.exploreHoveredIndex]
        ? exploreTrajectoriesState.exploreSamples[exploreTrajectoriesState.exploreHoveredIndex].position
        : undefined
    return doRenderExplorationTrajectories({
      samples: exploreTrajectoriesState.exploreSamples,
      hoveredIndex: exploreTrajectoriesState.exploreHoveredIndex,
      hoveredPosition: exploreTrajectoriesState.exploreMode === 'combinatorial' ? hoveredPosition : undefined,
      hoveredFromTrajectoryPath: exploreTrajectoriesState.exploreMode === 'combinatorial' ? exploreTrajectoriesState.exploreHoveredFromTrajectoryPath : undefined,
      exploreNodeId: exploreTrajectoriesState.exploreMode === 'combinatorial' ? exploreTrajectoriesState.exploreNodeId : undefined,
      unitsToPixels,
      jointNamesToShow: jointNamesToShow.length > 0 ? jointNamesToShow : undefined,
      exploreColormapEnabled,
      exploreColormapType,
      exploreCenter,
      exploreRadius,
      exploreMode: exploreTrajectoriesState.exploreMode
    })
  }

  const renderExplorationDots = (): React.ReactNode => {
    if (toolMode !== 'explore_node_trajectories' || !exploreTrajectoriesState.exploreSamples.length) return null
    const exploreCenter =
      exploreTrajectoriesState.exploreMode === 'combinatorial'
        ? exploreTrajectoriesState.exploreSecondCenter
        : exploreTrajectoriesState.exploreCenter
    const isCombinatorial = exploreTrajectoriesState.exploreMode === 'combinatorial'
    const samples = isCombinatorial
      ? deduplicateBySecondPosition(exploreTrajectoriesState.exploreSamples)
      : exploreTrajectoriesState.exploreSamples.map((s) => ({ position: s.position, valid: s.valid }))
    const hoveredPosition =
      exploreTrajectoriesState.exploreHoveredIndex != null && exploreTrajectoriesState.exploreSamples[exploreTrajectoriesState.exploreHoveredIndex]
        ? exploreTrajectoriesState.exploreSamples[exploreTrajectoriesState.exploreHoveredIndex].position
        : undefined
    return doRenderExplorationDots({
      samples,
      hoveredIndex: exploreTrajectoriesState.exploreHoveredIndex,
      hoveredPosition: isCombinatorial ? hoveredPosition : undefined,
      unitsToPixels,
      exploreColormapEnabled,
      exploreColormapType,
      exploreCenter,
      exploreRadius,
      exploreMode: exploreTrajectoriesState.exploreMode
    })
  }

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
      jointOutline,
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
    const trajectories = filterTrajectoriesForRendering(trajectoryData, pylinkDoc.meta.joints)
    return (
      <>
        {doRenderTrajectories({
          trajectories,
          trajectoryDotSize,
          trajectoryDotOutline,
          trajectoryDotOpacity,
          showTrajectoryStepNumbers,
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
    const linkEntries = Object.entries(pylinkDoc.meta.links)
      .filter(([, meta]) => meta.connects.length >= 2)
      .sort(([, a], [, b]) => {
        const za = (a as { zlevel?: number }).zlevel ?? 0
        const zb = (b as { zlevel?: number }).zlevel ?? 0
        return za - zb
      }) as [string, { connects: [string, string]; color?: string; isGround?: boolean; zlevel?: number }][]

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
      let baseLinkColor: string
      if (stretchingLinks.includes(linkName)) {
        baseLinkColor = '#ff0000'
      } else if (linkColorMode === 'single') {
        baseLinkColor = linkColorSingle
      } else if (linkColorMode === 'z-level') {
        baseLinkColor = linkMeta.color ?? getDefaultColor(linkMeta.zlevel ?? index)
      } else {
        baseLinkColor = isGround && !linkMeta.color
          ? '#7f7f7f'
          : (linkMeta.color || getDefaultColor(index))
      }

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
          linkTransparency,
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
        if (obj.mergedLinkName && (obj.mergedLinkOriginalStart != null || obj.mergedLinkOriginalEnd != null)) {
          const linkMeta = pylinkDoc.meta.links[obj.mergedLinkName]
          if (linkMeta) {
            const currentStart = getJointPosition(linkMeta.connects[0])
            const currentEnd = getJointPosition(linkMeta.connects[1])
            if (currentStart && currentEnd) {
              // Use trajectory[0] as transform origin when available so polygon (built at step 0) never
              // "moves away" at frame 0; avoids document vs trajectory[0] mismatch from merge/load.
              const traj0Start = trajectoryData?.trajectories?.[linkMeta.connects[0]]?.[0] as [number, number] | undefined
              const traj0End = trajectoryData?.trajectories?.[linkMeta.connects[1]]?.[0] as [number, number] | undefined
              const origStart = (traj0Start ?? obj.mergedLinkOriginalStart) as [number, number] | undefined
              const origEnd = (traj0End ?? obj.mergedLinkOriginalEnd) as [number, number] | undefined
              if (origStart && origEnd) {
                displayPoints = transformPolygonPoints(
                  obj.points,
                  origStart,
                  origEnd,
                  currentStart,
                  currentEnd
                )
              }
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
          mergedLinkName: obj.mergedLinkName,
          contained_links_valid: obj.contained_links_valid
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
          pointerEventsNoneForDrawPolygon: toolMode === 'draw_polygon',
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
        onPathClick: setSelectedPathId,
        trajectoryDotSize,
        trajectoryDotOpacity,
        trajectoryDotOutline
      })}
    </>
  )

  return {
    renderGrid,
    renderCanvases,
    renderDrawnObjects,
    renderLinks,
    renderPreviewLine,
    renderPolygonPreview,
    renderTargetPaths,
    renderPathPreview,
    renderExplorationTrajectories,
    renderTrajectories,
    renderExplorationDots,
    renderJoints,
    renderSelectionBox,
    renderMeasurementMarkers,
    renderMeasurementLine
  }
}
