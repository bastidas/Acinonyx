/**
 * useMoveGroup hook
 *
 * Encapsulates move-group drag logic: hit-test for start drag on selection,
 * translate joints and drawn objects on move, commit and update start positions on up.
 * Used by BuilderTab so canvas handlers can delegate to moveGroup.handleMouseDown/Move/Up.
 */

import { useCallback } from 'react'
import type { MoveGroupState } from '../../BuilderTools'

export type CanvasPoint = [number, number]

export interface UseMoveGroupParams {
  moveGroupState: MoveGroupState
  setMoveGroupState: React.Dispatch<React.SetStateAction<MoveGroupState>>
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
  drawnObjects: { objects: Array<{ id: string; points: CanvasPoint[]; mergedLinkName?: string }> }
  setDrawnObjects: React.Dispatch<React.SetStateAction<{ objects: unknown[]; selectedIds: string[] }>>
  translateGroupRigid: (
    jointNames: string[],
    originalPositions: Record<string, [number, number]>,
    dx: number,
    dy: number
  ) => void
  exitMoveGroupMode: () => void
  showStatus: (message: string, type?: string, duration?: number) => void
  triggerMechanismChange: () => void
}

export interface UseMoveGroupReturn {
  handleMouseDown: (event: React.MouseEvent<SVGSVGElement>, point: CanvasPoint) => boolean
  handleMouseMove: (event: React.MouseEvent<SVGSVGElement>, point: CanvasPoint) => boolean
  handleMouseUp: (event: React.MouseEvent<SVGSVGElement>) => boolean
  isDragging: boolean
}

const BBOX_PADDING = 0.5

/**
 * Hook that returns move-group drag handlers. When move group is active and user
 * clicks on the selection (joint, group link, bounding box, or drawn object), starts
 * drag; on move translates rigidly; on up commits and updates start positions.
 */
export function useMoveGroup(params: UseMoveGroupParams): UseMoveGroupReturn {
  const {
    moveGroupState,
    setMoveGroupState,
    getJointsWithPositions,
    getLinksWithPositions,
    getJointPosition,
    findNearestJoint,
    findNearestLink,
    getLinkConnects,
    drawnObjects,
    setDrawnObjects,
    translateGroupRigid,
    exitMoveGroupMode,
    showStatus,
    triggerMechanismChange
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
        setMoveGroupState(prev => ({
          ...prev,
          isDragging: true,
          dragStartPoint: clickPoint
        }))
        const itemCount = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
        showStatus(`Moving ${itemCount} items — drag to reposition`, 'action')
        return true
      }

      exitMoveGroupMode()
      return true
    },
    [
      moveGroupState.isActive,
      moveGroupState.joints,
      moveGroupState.drawnObjectIds,
      getJointsWithPositions,
      getLinksWithPositions,
      getJointPosition,
      findNearestJoint,
      findNearestLink,
      getLinkConnects,
      drawnObjects.objects,
      setMoveGroupState,
      exitMoveGroupMode,
      showStatus
    ]
  )

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGSVGElement>, currentPoint: CanvasPoint): boolean => {
      if (!moveGroupState.isDragging || !moveGroupState.dragStartPoint) {
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
          objects: (prev.objects as Array<{ id: string; points: CanvasPoint[]; mergedLinkName?: string }>).map(obj => {
            if (!moveGroupState.drawnObjectIds.includes(obj.id)) return obj
            if (obj.mergedLinkName) return obj
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
      moveGroupState.dragStartPoint,
      moveGroupState.joints,
      moveGroupState.startPositions,
      moveGroupState.drawnObjectIds,
      moveGroupState.drawnObjectStartPositions,
      translateGroupRigid,
      setDrawnObjects,
      showStatus
    ]
  )

  const handleMouseUp = useCallback(
    (event: React.MouseEvent<SVGSVGElement>): boolean => {
      if (!moveGroupState.isDragging) {
        return false
      }

      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moved ${totalItems} items`, 'success', 2000)
      setMoveGroupState(prev => ({
        ...prev,
        isDragging: false,
        dragStartPoint: null,
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
      return true
    },
    [
      moveGroupState.isDragging,
      moveGroupState.joints,
      moveGroupState.drawnObjectIds,
      setMoveGroupState,
      getJointPosition,
      drawnObjects.objects,
      showStatus,
      triggerMechanismChange
    ]
  )

  return {
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    isDragging: moveGroupState.isDragging
  }
}
