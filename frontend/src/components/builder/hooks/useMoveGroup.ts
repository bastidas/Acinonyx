/**
 * useMoveGroup hook
 *
 * Encapsulates move-group drag logic: hit-test for start drag on selection,
 * translate or rotate joints and drawn objects on move, commit and update start positions on up.
 * Used by BuilderTab so canvas handlers can delegate to moveGroup.handleMouseDown/Move/Up.
 */

import { useCallback } from 'react'
import type { MoveGroupState, ToolMode } from '../../BuilderTools'
import { isPointInPolygon } from '../../BuilderTools'

export type CanvasPoint = [number, number]

export interface UseMoveGroupParams {
  moveGroupState: MoveGroupState
  setMoveGroupState: React.Dispatch<React.SetStateAction<MoveGroupState>>
  toolMode: ToolMode
  getJointsWithPositions: () => Array<{ name: string; position: [number, number] | null }>
  getLinksWithPositions: () => Array<{ name: string; start: [number, number] | null; end: [number, number] | null }>
  getJointPosition: (jointName: string) => [number, number] | null
  findNearestJoint: (
    point: CanvasPoint,
    joints: Array<{ name: string; position: [number, number] | null }>,
    threshold?: number
  ) => { name: string; position: [number, number]; distance: number } | null
  findNearestLink: (
    point: CanvasPoint,
    links: Array<{ name: string; start: [number, number] | null; end: [number, number] | null }>,
    threshold?: number
  ) => { name: string; distance: number } | null
  /** Returns [connects0, connects1] for a link, or null. Used to check if link endpoints are in move group. */
  getLinkConnects: (linkName: string) => [string, string] | null
  drawnObjects: {
    objects: Array<{
      id: string
      points: CanvasPoint[]
      mergedLinkName?: string
      mergedLinkOriginalStart?: CanvasPoint
      mergedLinkOriginalEnd?: CanvasPoint
    }>
  }
  setDrawnObjects: React.Dispatch<React.SetStateAction<{ objects: unknown[]; selectedIds: string[] }>>
  translateGroupRigid: (
    jointNames: string[],
    originalPositions: Record<string, [number, number]>,
    dx: number,
    dy: number
  ) => void
  rotateGroupFromOriginal: (
    jointNames: string[],
    originalPositions: Record<string, [number, number]>,
    pivot: CanvasPoint,
    angleRad: number
  ) => void
  exitMoveGroupMode: () => void
  showStatus: (message: string, type?: string, duration?: number) => void
  triggerMechanismChange: () => void
  /** Reset animation to first frame when starting drag (avoids N/N flash). */
  resetAnimationToFirstFrame?: () => void
  /** After drag end, call to refresh polygon contained_links_valid (find-associated-polygons). */
  apiFindAssociatedPolygons?: () => Promise<unknown>
  /** After group drag ends: e.g. validate form–link associations from document positions. */
  onAfterMoveGroupDragEnd?: () => void
  /** After a committed group translate/rotate (mouse up); optional hook (e.g. analytics). */
  onGroupDragCommit?: () => void
  /** When a mechanism / form group drag starts (before first translate/rotate); capture one undo step for the whole gesture. */
  onGroupDragStart?: () => void
}

export interface UseMoveGroupReturn {
  handleMouseDown: (event: React.MouseEvent<SVGSVGElement>, point: CanvasPoint) => boolean
  handleMouseMove: (event: React.MouseEvent<SVGSVGElement>, point: CanvasPoint) => boolean
  handleMouseUp: (event: React.MouseEvent<SVGSVGElement>) => boolean
  isDragging: boolean
}

const BBOX_PADDING = 0.5

function angleDiff(a: number, b: number): number {
  let d = a - b
  while (d > Math.PI) d -= 2 * Math.PI
  while (d < -Math.PI) d += 2 * Math.PI
  return d
}

function rotatePointAboutPivot(p: CanvasPoint, pivot: CanvasPoint, angleRad: number): CanvasPoint {
  const cos = Math.cos(angleRad)
  const sin = Math.sin(angleRad)
  const dx = p[0] - pivot[0]
  const dy = p[1] - pivot[1]
  return [pivot[0] + dx * cos - dy * sin, pivot[1] + dx * sin + dy * cos]
}

function selectionCentroid(state: MoveGroupState): CanvasPoint | null {
  const pts: CanvasPoint[] = []
  for (const j of state.joints) {
    const p = state.startPositions[j]
    if (p) pts.push(p)
  }
  for (const id of state.drawnObjectIds) {
    const arr = state.drawnObjectStartPositions[id]
    if (arr) for (const p of arr) pts.push(p)
  }
  if (pts.length === 0) return null
  const sx = pts.reduce((s, p) => s + p[0], 0)
  const sy = pts.reduce((s, p) => s + p[1], 0)
  return [sx / pts.length, sy / pts.length]
}

/**
 * Hook that returns move-group drag handlers. When move group is active and user
 * clicks on the selection (joint, group link, bounding box, or drawn object), starts
 * drag; on move translates or (Mechanism Select + Alt) rotates rigidly; on up commits.
 */
export function useMoveGroup(params: UseMoveGroupParams): UseMoveGroupReturn {
  const {
    moveGroupState,
    setMoveGroupState,
    toolMode,
    getJointsWithPositions,
    getLinksWithPositions,
    getJointPosition,
    findNearestJoint,
    findNearestLink,
    getLinkConnects,
    drawnObjects,
    setDrawnObjects,
    translateGroupRigid,
    rotateGroupFromOriginal,
    exitMoveGroupMode,
    showStatus,
    triggerMechanismChange,
    resetAnimationToFirstFrame,
    apiFindAssociatedPolygons,
    onAfterMoveGroupDragEnd,
    onGroupDragCommit,
    onGroupDragStart
  } = params

  const handleMouseDown = useCallback(
    (event: React.MouseEvent<SVGSVGElement>, clickPoint: CanvasPoint): boolean => {
      const hasJoints = moveGroupState.joints.length > 0
      const hasDrawnObjects = moveGroupState.drawnObjectIds.length > 0
      if (!moveGroupState.isActive || (!hasJoints && !hasDrawnObjects)) {
        return false
      }

      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
      const nearestLink = findNearestLink(clickPoint, linksWithPositions)

      const clickedOnJoint = nearestJoint != null && moveGroupState.joints.includes(nearestJoint.name)

      const clickedOnGroupLink =
        hasJoints &&
        nearestLink != null &&
        (() => {
          const connects = getLinkConnects(nearestLink.name)
          if (!connects) return false
          return connects.every(j => moveGroupState.joints.includes(j))
        })()

      // Bounding box includes joints and/or drawn object points so polygons are moveable like links
      const clickedInBoundingBox = (() => {
        const jointPositions = hasJoints
          ? moveGroupState.joints
              .map(jointName => getJointPosition(jointName))
              .filter((pos): pos is [number, number] => pos !== null)
          : []
        const polygonPoints = hasDrawnObjects
          ? moveGroupState.drawnObjectIds.flatMap(id => {
              const obj = drawnObjects.objects.find((o: { id: string }) => o.id === id) as
                | { points: CanvasPoint[] }
                | undefined
              return obj?.points ?? []
            })
          : []
        const allPoints = [...jointPositions, ...polygonPoints]
        if (allPoints.length === 0) return false
        const minX = Math.min(...allPoints.map(p => p[0])) - BBOX_PADDING
        const maxX = Math.max(...allPoints.map(p => p[0])) + BBOX_PADDING
        const minY = Math.min(...allPoints.map(p => p[1])) - BBOX_PADDING
        const maxY = Math.max(...allPoints.map(p => p[1])) + BBOX_PADDING
        return (
          clickPoint[0] >= minX &&
          clickPoint[0] <= maxX &&
          clickPoint[1] >= minY &&
          clickPoint[1] <= maxY
        )
      })()

      const clickedOnDrawnObj = moveGroupState.drawnObjectIds.some(id => {
        const obj = drawnObjects.objects.find((o: { id: string }) => o.id === id) as
          | { id: string; points: CanvasPoint[] }
          | undefined
        if (!obj) return false
        if (obj.points.length >= 3) {
          return isPointInPolygon(clickPoint, obj.points)
        }
        const minX = Math.min(...obj.points.map(p => p[0]))
        const maxX = Math.max(...obj.points.map(p => p[0]))
        const minY = Math.min(...obj.points.map(p => p[1]))
        const maxY = Math.max(...obj.points.map(p => p[1]))
        return (
          clickPoint[0] >= minX &&
          clickPoint[0] <= maxX &&
          clickPoint[1] >= minY &&
          clickPoint[1] <= maxY
        )
      })

      if (clickedOnJoint || clickedOnGroupLink || clickedInBoundingBox || clickedOnDrawnObj) {
        const centroid = selectionCentroid(moveGroupState)
        const wantRotate =
          toolMode === 'mechanism_select' && event.altKey && centroid != null
        const dragMode = wantRotate ? 'rotate' : 'translate'
        const rotatePivot = wantRotate ? centroid : null
        const rotateRefAngle =
          wantRotate && rotatePivot
            ? Math.atan2(clickPoint[1] - rotatePivot[1], clickPoint[0] - rotatePivot[0])
            : null

        setMoveGroupState(prev => ({
          ...prev,
          isDragging: true,
          dragStartPoint: clickPoint,
          dragMode,
          rotatePivot,
          rotateRefAngle
        }))
        onGroupDragStart?.()
        resetAnimationToFirstFrame?.()
        const itemCount = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
        if (dragMode === 'rotate') {
          showStatus(`Rotating ${itemCount} items — Alt+drag`, 'action')
        } else {
          showStatus(`Moving ${itemCount} items — drag to reposition`, 'action')
        }
        return true
      }

      exitMoveGroupMode()
      return true
    },
    [
      moveGroupState.isActive,
      moveGroupState.joints,
      moveGroupState.drawnObjectIds,
      moveGroupState.startPositions,
      moveGroupState.drawnObjectStartPositions,
      toolMode,
      getJointsWithPositions,
      getLinksWithPositions,
      getJointPosition,
      findNearestJoint,
      findNearestLink,
      getLinkConnects,
      drawnObjects.objects,
      setMoveGroupState,
      exitMoveGroupMode,
      showStatus,
      resetAnimationToFirstFrame,
      onGroupDragStart
    ]
  )

  const handleMouseMove = useCallback(
    (_event: React.MouseEvent<SVGSVGElement>, currentPoint: CanvasPoint): boolean => {
      if (!moveGroupState.isDragging) {
        return false
      }

      if (moveGroupState.dragMode === 'rotate') {
        const pivot = moveGroupState.rotatePivot
        const refA = moveGroupState.rotateRefAngle
        if (pivot == null || refA == null) {
          return false
        }

        const curA = Math.atan2(currentPoint[1] - pivot[1], currentPoint[0] - pivot[0])
        const angleRad = angleDiff(curA, refA)

        rotateGroupFromOriginal(
          moveGroupState.joints,
          moveGroupState.startPositions,
          pivot,
          angleRad
        )

        if (moveGroupState.drawnObjectIds.length > 0) {
          setDrawnObjects(prev => ({
            ...prev,
            objects: (prev.objects as Array<{
              id: string
              points: CanvasPoint[]
              mergedLinkName?: string
              mergedLinkOriginalStart?: CanvasPoint
              mergedLinkOriginalEnd?: CanvasPoint
            }>).map(obj => {
              if (!moveGroupState.drawnObjectIds.includes(obj.id)) return obj
              const canUseLinkTransform =
                obj.mergedLinkName != null &&
                obj.mergedLinkOriginalStart != null &&
                obj.mergedLinkOriginalEnd != null &&
                getLinkConnects(obj.mergedLinkName) != null
              if (canUseLinkTransform) return obj
              const originalPoints = moveGroupState.drawnObjectStartPositions[obj.id]
              if (originalPoints) {
                return {
                  ...obj,
                  points: originalPoints.map(p => rotatePointAboutPivot(p, pivot, angleRad))
                }
              }
              return obj
            })
          }))
        }

        const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
        const deg = (angleRad * 180) / Math.PI
        showStatus(`Rotating ${totalItems} items (${deg.toFixed(1)}°)`, 'action')
        return true
      }

      if (!moveGroupState.dragStartPoint) {
        return false
      }

      const dx = currentPoint[0] - moveGroupState.dragStartPoint[0]
      const dy = currentPoint[1] - moveGroupState.dragStartPoint[1]

      translateGroupRigid(
        moveGroupState.joints,
        moveGroupState.startPositions,
        dx,
        dy
      )

      if (moveGroupState.drawnObjectIds.length > 0) {
        setDrawnObjects(prev => ({
          ...prev,
          objects: (prev.objects as Array<{
            id: string
            points: CanvasPoint[]
            mergedLinkName?: string
            mergedLinkOriginalStart?: CanvasPoint
            mergedLinkOriginalEnd?: CanvasPoint
          }>).map(obj => {
            if (!moveGroupState.drawnObjectIds.includes(obj.id)) return obj
            const canUseLinkTransform =
              obj.mergedLinkName != null &&
              obj.mergedLinkOriginalStart != null &&
              obj.mergedLinkOriginalEnd != null &&
              getLinkConnects(obj.mergedLinkName) != null
            if (canUseLinkTransform) return obj
            const originalPoints = moveGroupState.drawnObjectStartPositions[obj.id]
            if (originalPoints) {
              return {
                ...obj,
                points: originalPoints.map(p => [p[0] + dx, p[1] + dy] as [number, number])
              }
            }
            return obj
          })
        }))
      }

      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moving ${totalItems} items (Δ${dx.toFixed(1)}, ${dy.toFixed(1)})`, 'action')
      return true
    },
    [
      moveGroupState.isDragging,
      moveGroupState.dragMode,
      moveGroupState.dragStartPoint,
      moveGroupState.rotatePivot,
      moveGroupState.rotateRefAngle,
      moveGroupState.joints,
      moveGroupState.startPositions,
      moveGroupState.drawnObjectIds,
      moveGroupState.drawnObjectStartPositions,
      translateGroupRigid,
      rotateGroupFromOriginal,
      getLinkConnects,
      setDrawnObjects,
      showStatus
    ]
  )

  const handleMouseUp = useCallback(
    (_event: React.MouseEvent<SVGSVGElement>): boolean => {
      if (!moveGroupState.isDragging) {
        return false
      }

      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      const wasRotate = moveGroupState.dragMode === 'rotate'
      showStatus(
        wasRotate ? `Rotated ${totalItems} items` : `Moved ${totalItems} items`,
        'success',
        2000
      )
      setMoveGroupState(prev => ({
        ...prev,
        isDragging: false,
        dragStartPoint: null,
        dragMode: 'translate',
        rotatePivot: null,
        rotateRefAngle: null,
        startPositions: Object.fromEntries(
          prev.joints.map(jointName => {
            const pos = getJointPosition(jointName)
            return [jointName, pos ?? [0, 0]]
          })
        ),
        drawnObjectStartPositions: Object.fromEntries(
          prev.drawnObjectIds.map(id => {
            const obj = drawnObjects.objects.find((o: { id: string }) => o.id === id) as
              | { id: string; points: CanvasPoint[] }
              | undefined
            return [id, obj ? [...obj.points] : []]
          })
        )
      }))
      triggerMechanismChange()
      onGroupDragCommit?.()
      setTimeout(() => {
        onAfterMoveGroupDragEnd?.()
        void apiFindAssociatedPolygons?.()
      }, 0)
      return true
    },
    [
      moveGroupState.isDragging,
      moveGroupState.dragMode,
      moveGroupState.joints,
      moveGroupState.drawnObjectIds,
      setMoveGroupState,
      getJointPosition,
      drawnObjects.objects,
      showStatus,
      triggerMechanismChange,
      apiFindAssociatedPolygons,
      onAfterMoveGroupDragEnd,
      onGroupDragCommit
    ]
  )

  return {
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    isDragging: moveGroupState.isDragging
  }
}
