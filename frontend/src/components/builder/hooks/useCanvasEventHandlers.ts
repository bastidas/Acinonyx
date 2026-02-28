/**
 * useCanvasEventHandlers
 *
 * Builds the six canvas mouse handlers (mouseDown, mouseMove, mouseUp, mouseLeave, click, doubleClick)
 * for BuilderCanvasArea. Isolates canvas event logic from BuilderTab.
 */

import { useCallback } from 'react'
import { flushSync } from 'react-dom'
import { getToolHandler } from '../toolHandlers'
import type { ToolContext } from '../toolHandlers/types'
import type { ToolMode, DrawnObjectsState, StatusType } from '../../BuilderTools'
import type { LinkMetaData } from '../../BuilderTools'
import { findConnectedMechanism, JOINT_SNAP_THRESHOLD } from '../../BuilderTools'
import type { RefObject } from 'react'

export interface UseCanvasEventHandlersParams {
  canvasRef: RefObject<SVGSVGElement | HTMLDivElement | null>
  viewport: {
    handlePanStart: (event: React.MouseEvent) => boolean
    handlePanMove: (event: React.MouseEvent) => void
    handlePanEnd: () => void
    isPanningRef: React.MutableRefObject<boolean>
  }
  screenToUnit: (pixelX: number, pixelY: number) => [number, number]
  toolMode: ToolMode
  toolContext: ToolContext
  moveGroup: {
    handleMouseDown: (event: React.MouseEvent<SVGSVGElement>, point: [number, number]) => boolean
    handleMouseMove: (event: React.MouseEvent<SVGSVGElement>, point: [number, number]) => boolean
    handleMouseUp: (event: React.MouseEvent<SVGSVGElement>) => boolean
  }
  enterEditMode: () => void
  animation: {
    animationState: { isAnimating: boolean }
    pauseAnimation: () => void
  }
  showStatus: (message: string, type?: StatusType, duration?: number) => void
  dragState: { isDragging: boolean }
  groupSelectionState: { isSelecting: boolean }
  getJointsWithPositions: () => Array<{ name: string; position: [number, number] | null }>
  getLinksWithPositions: () => Array<{
    name: string
    start: [number, number] | null
    end: [number, number] | null
  }>
  findNearestJoint: (
    point: [number, number],
    joints: Array<{ name: string; position: [number, number] | null }>,
    threshold?: number
  ) => { name: string; position: [number, number]; distance: number } | null
  findNearestLink: (
    point: [number, number],
    links: Array<{
      name: string
      start: [number, number] | null
      end: [number, number] | null
    }>,
    threshold?: number
  ) => { name: string; distance: number } | null
  drawnObjects: { objects: Array<{ id: string; mergedLinkName?: string }> }
  setDrawnObjects: React.Dispatch<React.SetStateAction<DrawnObjectsState>>
  setSelectedJoints: (value: string[] | ((prev: string[]) => string[])) => void
  setSelectedLinks: (value: string[] | ((prev: string[]) => string[])) => void
  enterMoveGroupMode: (jointNames: string[], drawnObjectIds?: string[]) => void
  exitMoveGroupMode: () => void
  pylinkDoc: { meta: { links: Record<string, { connects: string[] }> }; pylinkage: { joints: unknown[] } }
}

export interface UseCanvasEventHandlersReturn {
  handleCanvasMouseDown: (event: React.MouseEvent<SVGSVGElement>) => void
  handleCanvasMouseMove: (event: React.MouseEvent<SVGSVGElement>) => void
  handleCanvasMouseUp: (event: React.MouseEvent<SVGSVGElement>) => void
  handleCanvasMouseLeave: (event: React.MouseEvent<SVGSVGElement>) => void
  handleCanvasClick: (event: React.MouseEvent<SVGSVGElement>) => void
  handleCanvasDoubleClick: (event: React.MouseEvent<SVGSVGElement>) => void
}

export function useCanvasEventHandlers(
  params: UseCanvasEventHandlersParams
): UseCanvasEventHandlersReturn {
  const {
    canvasRef,
    viewport,
    screenToUnit,
    toolMode,
    toolContext,
    moveGroup,
    enterEditMode,
    animation,
    showStatus,
    dragState,
    groupSelectionState,
    getJointsWithPositions,
    getLinksWithPositions,
    findNearestJoint,
    findNearestLink,
    drawnObjects,
    setDrawnObjects,
    setSelectedJoints,
    setSelectedLinks,
    enterMoveGroupMode,
    exitMoveGroupMode,
    pylinkDoc
  } = params

  const handleCanvasMouseDown = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      if (!canvasRef.current) return
      if (viewport.handlePanStart(event)) return
      if (animation.animationState.isAnimating) {
        animation.pauseAnimation()
        showStatus('Animation paused for editing', 'info', 1500)
      }
      const rect = canvasRef.current.getBoundingClientRect()
      const pixelX = event.clientX - rect.left
      const pixelY = event.clientY - rect.top
      const clickPoint = screenToUnit(pixelX, pixelY)
      if (moveGroup.handleMouseDown(event, clickPoint)) {
        flushSync(enterEditMode)
        return
      }
      const point = toolContext.getPointFromEvent(event)
      if (point) {
        const handler = getToolHandler(toolMode)
        const handled = handler.onMouseDown?.(event, point, toolContext)
        if (handled) {
          flushSync(enterEditMode)
          return
        }
      }
    },
    [
      toolMode,
      toolContext,
      moveGroup,
      viewport,
      screenToUnit,
      enterEditMode,
      animation.animationState.isAnimating,
      animation.pauseAnimation,
      showStatus
    ]
  )

  const handleCanvasMouseMove = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      if (!canvasRef.current) return
      if (viewport.isPanningRef.current) {
        viewport.handlePanMove(event)
        return
      }
      const rect = canvasRef.current.getBoundingClientRect()
      const pixelX = event.clientX - rect.left
      const pixelY = event.clientY - rect.top
      const currentPoint = screenToUnit(pixelX, pixelY)
      if (moveGroup.handleMouseMove(event, currentPoint)) return
      const handler = getToolHandler(toolMode)
      if (handler.onMouseMove?.(event, currentPoint, toolContext)) return
    },
    [toolMode, toolContext, moveGroup, viewport, screenToUnit]
  )

  const handleCanvasMouseUp = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      viewport.handlePanEnd()
      if (moveGroup.handleMouseUp(event)) return
      const handler = getToolHandler(toolMode)
      if (handler.onMouseUp?.(event, toolContext)) return
    },
    [toolMode, toolContext, moveGroup, viewport]
  )

  const handleCanvasMouseLeave = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      viewport.handlePanEnd()
      const handler = getToolHandler(toolMode)
      if (handler.onMouseLeave?.(event, toolContext)) return
    },
    [toolMode, toolContext, viewport]
  )

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      if (dragState.isDragging || groupSelectionState.isSelecting) return
      if (!canvasRef.current) return
      const rect = canvasRef.current.getBoundingClientRect()
      const pixelX = event.clientX - rect.left
      const pixelY = event.clientY - rect.top
      const clickPoint = screenToUnit(pixelX, pixelY)
      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
      const linkThreshold = toolMode === 'merge' ? 8.0 : JOINT_SNAP_THRESHOLD
      const nearestLink = findNearestLink(clickPoint, linksWithPositions, linkThreshold)
      const handler = getToolHandler(toolMode)
      if (handler.onClick?.(event, clickPoint, toolContext)) return
      if (toolMode === 'mechanism_select') {
        const findMergedDrawnObjects = (linkNames: string[]): string[] => {
          return drawnObjects.objects
            .filter(obj => obj.mergedLinkName && linkNames.includes(obj.mergedLinkName))
            .map(obj => obj.id)
        }
        if (nearestJoint) {
          const mechanism = findConnectedMechanism(nearestJoint.name, pylinkDoc.meta.links as Record<string, LinkMetaData>)
          setSelectedJoints(mechanism.joints)
          setSelectedLinks(mechanism.links)
          const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
          setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
          enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
        } else if (nearestLink) {
          const linkMeta = pylinkDoc.meta.links[nearestLink.name]
          if (linkMeta && linkMeta.connects.length > 0) {
            const mechanism = findConnectedMechanism(linkMeta.connects[0], pylinkDoc.meta.links as Record<string, LinkMetaData>)
            setSelectedJoints(mechanism.joints)
            setSelectedLinks(mechanism.links)
            const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
            setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
            enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
          }
        } else {
          setSelectedJoints([])
          setSelectedLinks([])
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
          exitMoveGroupMode()
          showStatus('Click on a joint or link to select its mechanism', 'info', 1500)
        }
      }
    },
    [
      toolMode,
      toolContext,
      dragState.isDragging,
      groupSelectionState.isSelecting,
      getJointsWithPositions,
      getLinksWithPositions,
      pylinkDoc.meta.links,
      drawnObjects.objects,
      setDrawnObjects,
      setSelectedJoints,
      setSelectedLinks,
      enterMoveGroupMode,
      exitMoveGroupMode,
      showStatus,
      screenToUnit
    ]
  )

  const handleCanvasDoubleClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const point = toolContext.getPointFromEvent(event)
      if (point) {
        const handler = getToolHandler(toolMode)
        if (handler.onDoubleClick?.(event, point, toolContext)) return
      }
    },
    [toolMode, toolContext]
  )

  return {
    handleCanvasMouseDown,
    handleCanvasMouseMove,
    handleCanvasMouseUp,
    handleCanvasMouseLeave,
    handleCanvasClick,
    handleCanvasDoubleClick
  }
}
