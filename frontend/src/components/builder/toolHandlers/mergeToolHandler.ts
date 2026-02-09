/**
 * Merge tool handler
 *
 * Handles merge mode: select polygon then link (or link then polygon) to merge
 * a polygon with an enclosed link. Extracted from BuilderTab handleCanvasClick.
 *
 * Shared API for element clicks (Option A):
 * - handleMergeLinkClick(context, linkName, position0, position1): call when user clicks a link in merge mode.
 * - handleMergePolygonClick(context, params): call when user clicks a polygon in merge mode (select, complete merge, switch, or unmerge).
 * Same contract: mergePolygonState/setMergePolygonState, setDrawnObjects, setSelectedLinks, showStatus.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialMergePolygonState } from '../../BuilderTools'

/** Params for polygon click in merge mode (select, complete merge, switch polygon, or unmerge). */
export interface HandleMergePolygonClickParams {
  polygonId: string
  polygonName: string
  /** True if polygon is already merged — perform unmerge using current link positions. */
  isUnmerge?: boolean
  mergedLinkName?: string
  mergedLinkOriginalStart?: CanvasPoint
  mergedLinkOriginalEnd?: CanvasPoint
  /** Polygon points (for unmerge: current display points before applying final transform). */
  polygonPoints: CanvasPoint[]
}

/**
 * Handle link click in merge mode. Call from renderLinks when user clicks a link.
 * Returns true if the click was handled (merge selection / complete / switch).
 */
export function handleMergeLinkClick(
  context: ToolContext,
  linkName: string,
  position0: CanvasPoint,
  position1: CanvasPoint,
  linkColor?: string
): boolean {
  if (context.toolMode !== 'merge') return false

  const { mergePolygonState } = context

  if (mergePolygonState.step === 'idle' || mergePolygonState.step === 'awaiting_selection') {
    context.setMergePolygonState({
      step: 'link_selected',
      selectedPolygonId: null,
      selectedLinkName: linkName
    })
    context.setSelectedLinks([linkName])
    context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [] }))
    context.showStatus(`Selected link "${linkName}" — click a polygon to merge with`, 'action')
    return true
  }

  if (mergePolygonState.step === 'polygon_selected') {
    const polygonId = mergePolygonState.selectedPolygonId!
    const polygon = context.drawnObjects.objects.find((obj: { id: string }) => obj.id === polygonId) as { id: string; name?: string; points: CanvasPoint[] } | undefined
    if (polygon?.points && context.areLinkEndpointsInPolygon(position0, position1, polygon.points)) {
      const color = linkColor ?? context.getDefaultColor(0)
      context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({
        ...prev,
        objects: (prev.objects as Array<Record<string, unknown>>).map(obj => {
          if (obj.id === polygonId) {
            return {
              ...obj,
              mergedLinkName: linkName,
              mergedLinkOriginalStart: position0,
              mergedLinkOriginalEnd: position1,
              fillColor: color,
              fillOpacity: 0.25,
              strokeColor: color
            }
          }
          return obj
        }),
        selectedIds: []
      }))
      context.showStatus(`✓ Merged polygon "${polygon.name ?? polygonId}" with link "${linkName}"`, 'success', 3000)
      context.setMergePolygonState(initialMergePolygonState)
      context.setSelectedLinks([])
    } else if (polygon) {
      context.showStatus(`✗ Failed: Link "${linkName}" endpoints are not inside polygon "${polygon.name ?? polygonId}"`, 'error', 3500)
    }
    return true
  }

  if (mergePolygonState.step === 'link_selected' && linkName !== mergePolygonState.selectedLinkName) {
    context.setMergePolygonState({
      step: 'link_selected',
      selectedPolygonId: null,
      selectedLinkName: linkName
    })
    context.setSelectedLinks([linkName])
    context.showStatus(`Switched to link "${linkName}" — click a polygon to merge with`, 'action')
  }
  return true
}

/**
 * Handle polygon click in merge mode. Call from renderDrawnObjects when user clicks a polygon.
 * Handles: unmerge (if isUnmerge), select polygon, complete merge with selected link, or switch polygon.
 * Returns true if the click was handled.
 */
export function handleMergePolygonClick(context: ToolContext, params: HandleMergePolygonClickParams): boolean {
  if (context.toolMode !== 'merge') return false

  const { polygonId, polygonName, isUnmerge, mergedLinkName, mergedLinkOriginalStart, mergedLinkOriginalEnd, polygonPoints } = params
  const { mergePolygonState } = context

  if (isUnmerge && mergedLinkName) {
    let finalPoints = polygonPoints
    if (mergedLinkOriginalStart && mergedLinkOriginalEnd) {
      const linkMeta = context.getLinkMeta(mergedLinkName)
      if (linkMeta) {
        const currentStart = context.getJointPosition(linkMeta.connects[0])
        const currentEnd = context.getJointPosition(linkMeta.connects[1])
        if (currentStart && currentEnd) {
          finalPoints = context.transformPolygonPoints(
            polygonPoints,
            mergedLinkOriginalStart,
            mergedLinkOriginalEnd,
            currentStart,
            currentEnd
          )
        }
      }
    }
    context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({
      ...prev,
      objects: (prev.objects as Array<Record<string, unknown>>).map(obj => {
        if (obj.id === polygonId) {
          return {
            ...obj,
            points: finalPoints,
            mergedLinkName: undefined,
            mergedLinkOriginalStart: undefined,
            mergedLinkOriginalEnd: undefined,
            fillColor: 'rgba(156, 39, 176, 0.15)',
            fillOpacity: 0.15,
            strokeColor: '#9c27b0'
          }
        }
        return obj
      })
    }))
    context.showStatus(`✓ Unmerged polygon "${polygonName}" from link "${mergedLinkName}"`, 'success', 3000)
    context.setMergePolygonState(initialMergePolygonState)
    context.setSelectedLinks([])
    return true
  }

  if (mergePolygonState.step === 'idle' || mergePolygonState.step === 'awaiting_selection') {
    context.setMergePolygonState({
      step: 'polygon_selected',
      selectedPolygonId: polygonId,
      selectedLinkName: null
    })
    context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [polygonId] }))
    context.setSelectedLinks([])
    context.showStatus(`Selected polygon "${polygonName}" — click a link to merge with`, 'action')
    return true
  }

  if (mergePolygonState.step === 'link_selected') {
    const linkName = mergePolygonState.selectedLinkName!
    const linkMeta = context.getLinkMeta(linkName)
    if (linkMeta) {
      const startPos = context.getJointPosition(linkMeta.connects[0])
      const endPos = context.getJointPosition(linkMeta.connects[1])
      if (startPos && endPos && context.areLinkEndpointsInPolygon(startPos, endPos, polygonPoints)) {
        const linkColor = linkMeta.color ?? context.getDefaultColor(0)
        context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({
          ...prev,
          objects: (prev.objects as Array<Record<string, unknown>>).map(o => {
            if (o.id === polygonId) {
              return {
                ...o,
                mergedLinkName: linkName,
                mergedLinkOriginalStart: startPos,
                mergedLinkOriginalEnd: endPos,
                fillColor: linkColor,
                fillOpacity: 0.25,
                strokeColor: linkColor
              }
            }
            return o
          }),
          selectedIds: []
        }))
        context.showStatus(`✓ Merged polygon "${polygonName}" with link "${linkName}"`, 'success', 3000)
        context.setMergePolygonState(initialMergePolygonState)
        context.setSelectedLinks([])
      } else {
        context.showStatus(`✗ Failed: Link "${linkName}" endpoints are not inside polygon "${polygonName}"`, 'error', 3500)
      }
    }
    return true
  }

  if (mergePolygonState.step === 'polygon_selected' && polygonId !== mergePolygonState.selectedPolygonId) {
    context.setMergePolygonState({
      step: 'polygon_selected',
      selectedPolygonId: polygonId,
      selectedLinkName: null
    })
    context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [polygonId] }))
    context.showStatus(`Switched to polygon "${polygonName}" — click a link to merge with`, 'action')
  }
  return true
}

/** Internal: perform merge (polygon + link) and reset state. */
function completeMerge(context: ToolContext, polygonId: string, linkName: string): boolean {
  const polygon = context.drawnObjects.objects.find((obj: { id: string }) => obj.id === polygonId) as { id: string; name?: string; points: CanvasPoint[] } | undefined
  const linkMeta = context.getLinkMeta(linkName)

  if (!polygon || !linkMeta) {
    context.showStatus('Error: Could not find polygon or link', 'error', 2000)
    context.setMergePolygonState(initialMergePolygonState)
    return false
  }

  const startPos = context.getJointPosition(linkMeta.connects[0])
  const endPos = context.getJointPosition(linkMeta.connects[1])

  if (!startPos || !endPos) {
    context.showStatus('Error: Could not get link endpoint positions', 'error', 2000)
    return false
  }

  if (!context.areLinkEndpointsInPolygon(startPos, endPos, polygon.points)) {
    context.showStatus(`Link "${linkName}" endpoints are not inside the polygon. Both ends must be enclosed.`, 'warning', 3500)
    return false
  }

  const linkColor = linkMeta.color ?? context.getDefaultColor(0)

  context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({
    ...prev,
    objects: (prev.objects as Array<Record<string, unknown>>).map(obj => {
      if (obj.id === polygonId) {
        return {
          ...obj,
          mergedLinkName: linkName,
          mergedLinkOriginalStart: startPos,
          mergedLinkOriginalEnd: endPos,
          fillColor: linkColor,
          fillOpacity: 0.25,
          strokeColor: linkColor
        }
      }
      return obj
    }),
    selectedIds: []
  }))

  context.showStatus(`Merged polygon "${polygon.name ?? polygonId}" with link "${linkName}"`, 'success', 3000)
  context.setMergePolygonState(initialMergePolygonState)
  context.setSelectedLinks([])
  return true
}

export const mergeToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'merge') return false

    const jointsWithPositions = context.getJointsWithPositions()
    const linksWithPositions = context.getLinksWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    const nearestLink = context.findNearestLink(point, linksWithPositions, context.mergeLinkThreshold)
    const clickPoint = point

    const findClickedPolygon = () => {
      return context.drawnObjects.objects.find(obj => {
        if (obj.type !== 'polygon' || obj.points.length < 3) return false
        if (obj.mergedLinkName) return false
        if (context.isPointInPolygon(clickPoint, obj.points)) return true
        for (let i = 0; i < obj.points.length; i++) {
          const p1 = obj.points[i]
          const p2 = obj.points[(i + 1) % obj.points.length]
          const lineLen = context.calculateDistance(p1, p2)
          if (lineLen === 0) continue
          const t = Math.max(0, Math.min(1, ((clickPoint[0] - p1[0]) * (p2[0] - p1[0]) + (clickPoint[1] - p1[1]) * (p2[1] - p1[1])) / (lineLen * lineLen)))
          const projX = p1[0] + t * (p2[0] - p1[0])
          const projY = p1[1] + t * (p2[1] - p1[1])
          const dist = context.calculateDistance(clickPoint, [projX, projY])
          if (dist < 0.5) return true
        }
        return false
      })
    }

    const { mergePolygonState } = context

    if (mergePolygonState.step === 'idle' || mergePolygonState.step === 'awaiting_selection') {
      if (nearestLink) {
        context.setMergePolygonState({
          step: 'link_selected',
          selectedPolygonId: null,
          selectedLinkName: nearestLink.name
        })
        context.setSelectedLinks([nearestLink.name])
        context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [] }))
        context.showStatus(`Selected link "${nearestLink.name}" — now click a polygon to merge`, 'action')
        return true
      }

      const clickedPolygon = findClickedPolygon()
      if (clickedPolygon) {
        context.setMergePolygonState({
          step: 'polygon_selected',
          selectedPolygonId: clickedPolygon.id,
          selectedLinkName: null
        })
        context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [clickedPolygon.id] }))
        context.setSelectedLinks([])
        context.showStatus(`Selected polygon "${(clickedPolygon as { name?: string }).name ?? clickedPolygon.id}" — now click a link to merge`, 'action')
        return true
      }

      const mergedPolygon = context.drawnObjects.objects.find(obj => {
        if (obj.type !== 'polygon' || obj.points.length < 3) return false
        if (!obj.mergedLinkName) return false
        return context.isPointInPolygon(clickPoint, obj.points)
      })
      if (mergedPolygon) {
        context.showStatus(`Polygon "${(mergedPolygon as { name?: string }).name ?? mergedPolygon.id}" is already merged with link "${mergedPolygon.mergedLinkName}"`, 'info', 2500)
      } else {
        context.showStatus('Select a link or a polygon to begin merge', 'info', 2000)
      }
      return true
    }

    if (mergePolygonState.step === 'polygon_selected') {
      if (nearestLink) {
        const success = completeMerge(context, mergePolygonState.selectedPolygonId!, nearestLink.name)
        if (!success) {
          context.showStatus('Link endpoints must be inside the polygon. Try another link.', 'warning', 2500)
        }
        return true
      }

      const clickedPolygon = findClickedPolygon()
      if (clickedPolygon && clickedPolygon.id !== mergePolygonState.selectedPolygonId) {
        context.setMergePolygonState({
          step: 'polygon_selected',
          selectedPolygonId: clickedPolygon.id,
          selectedLinkName: null
        })
        context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [clickedPolygon.id] }))
        context.showStatus(`Switched to polygon "${(clickedPolygon as { name?: string }).name ?? clickedPolygon.id}" — now click a link to merge`, 'action')
      } else {
        context.showStatus('Click a link to merge with the selected polygon', 'info', 2000)
      }
      return true
    }

    if (mergePolygonState.step === 'link_selected') {
      if (nearestLink && nearestLink.name !== mergePolygonState.selectedLinkName) {
        context.setMergePolygonState({
          step: 'link_selected',
          selectedPolygonId: null,
          selectedLinkName: nearestLink.name
        })
        context.setSelectedLinks([nearestLink.name])
        context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => ({ ...prev, selectedIds: [] }))
        context.showStatus(`Switched to link "${nearestLink.name}" — now click a polygon to merge`, 'action')
        return true
      }

      const clickedPolygon = findClickedPolygon()
      if (clickedPolygon) {
        const success = completeMerge(context, clickedPolygon.id, mergePolygonState.selectedLinkName!)
        if (!success) {
          context.showStatus('Link endpoints must be inside the polygon. Try another polygon.', 'warning', 2500)
        }
        return true
      }

      context.showStatus('Click inside a polygon to merge with the selected link', 'info', 2000)
      return true
    }

    return true
  }
}
