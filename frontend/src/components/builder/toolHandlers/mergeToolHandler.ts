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
    context.showStatus(`Selected link "${linkName}" — click a polygon form to merge with`, 'action')
    return true
  }

  if (mergePolygonState.step === 'polygon_selected') {
    const polygonId = mergePolygonState.selectedPolygonId!
    const polygon = context.drawnObjects.objects.find((obj: { id: string }) => obj.id === polygonId) as { id: string; name?: string; points: CanvasPoint[] } | undefined
    if (!polygon?.points || polygon.points.length < 3) {
      context.showStatus('Invalid polygon', 'error', 2000)
      return true
    }
    if (!context.apiMergePolygon) {
      context.showStatus('Merge requires backend connection', 'error', 3000)
      return true
    }
    context.apiMergePolygon({ polygonId, polygonPoints: polygon.points }).then(data => {
      if (data.status === 'success' && data.polygon?.contained_links?.length) {
        context.showStatus(`✓ Merged polygon with ${data.polygon.contained_links.length} link(s)`, 'success', 3000)
      } else if (data.status === 'success') {
        context.showStatus('No fully bounded links inside polygon', 'warning', 3000)
      } else {
        context.showStatus(data.message ?? 'Merge failed', 'error', 3000)
      }
      context.setMergePolygonState(initialMergePolygonState)
      context.setSelectedLinks([])
    })
    return true
  }

  if (mergePolygonState.step === 'link_selected' && linkName !== mergePolygonState.selectedLinkName) {
    context.setMergePolygonState({
      step: 'link_selected',
      selectedPolygonId: null,
      selectedLinkName: linkName
    })
    context.setSelectedLinks([linkName])
    context.showStatus(`Switched to link "${linkName}" — click a polygon form to merge with`, 'action')
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
            contained_links: undefined,
            contained_links_valid: undefined,
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
    context.showStatus(`Selected polygon "${polygonName}" — click a link or another polygon form to merge with`, 'action')
    return true
  }

  if (mergePolygonState.step === 'link_selected') {
    const selectedLinkName = mergePolygonState.selectedLinkName ?? undefined
    if (polygonPoints.length < 3) {
      context.showStatus('Invalid polygon', 'error', 2000)
      return true
    }
    if (!context.apiMergePolygon) {
      context.showStatus('Merge requires backend connection', 'error', 3000)
      return true
    }
    context.apiMergePolygon({ polygonId, polygonPoints, selectedLinkName }).then(data => {
      const p = data.polygon
      const selectedFullyInside = p?.selected_link_fully_inside
      if (selectedLinkName && selectedFullyInside === false) {
        context.showStatus(`Link "${selectedLinkName}" is not fully inside polygon`, 'error', 3500)
      } else if (data.status === 'success' && p?.contained_links?.length) {
        context.showStatus(`✓ Merged polygon "${polygonName}" with ${p.contained_links.length} link(s)`, 'success', 3000)
      } else if (data.status === 'success') {
        context.showStatus('No fully bounded links inside polygon', 'warning', 3000)
      } else {
        context.showStatus(data.message ?? 'Merge failed', 'error', 3000)
      }
      context.setMergePolygonState(initialMergePolygonState)
      context.setSelectedLinks([])
    })
    return true
  }

  if (mergePolygonState.step === 'polygon_selected' && polygonId !== mergePolygonState.selectedPolygonId) {
    const polygonIdA = mergePolygonState.selectedPolygonId!
    const objA = context.drawnObjects.objects.find((o: { id: string }) => o.id === polygonIdA) as {
      id: string
      name?: string
      points: CanvasPoint[]
      fillColor?: string
      strokeColor?: string
      z_level?: number
      [k: string]: unknown
    } | undefined
    const polygonB = context.drawnObjects.objects.find((o: { id: string }) => o.id === polygonId) as {
      id: string
      name?: string
      points: CanvasPoint[]
    } | undefined
    if (!objA?.points || objA.points.length < 3) {
      context.showStatus('Could not find selected polygon', 'error', 2000)
      return true
    }
    if (!polygonB?.points || polygonB.points.length < 3) {
      context.showStatus('Invalid polygon to merge', 'error', 2000)
      return true
    }
    if (!context.apiMergeTwoPolygons) {
      context.showStatus('Merge two polygons requires backend connection', 'error', 3000)
      return true
    }
    context.apiMergeTwoPolygons({
      polygonIdA,
      polygonPointsA: objA.points,
      polygonIdB: polygonId,
      polygonPointsB: polygonB.points
    }).then(data => {
      if (data.status !== 'success' || !data.merged_polygon) {
        context.showStatus(data.message ?? 'Merge two polygons failed', 'error', 3000)
        return
      }
      const mp = data.merged_polygon
      const contained = mp.contained_links ?? []
      const primary = contained[0]
      let mergedLinkOriginalStart: CanvasPoint | undefined
      let mergedLinkOriginalEnd: CanvasPoint | undefined
      if (primary) {
        const linkMeta = context.getLinkMeta(primary) as { connects?: [string, string] } | null
        if (linkMeta?.connects) {
          const p0 = context.getJointPosition(linkMeta.connects[0])
          const p1 = context.getJointPosition(linkMeta.connects[1])
          if (p0 && p1) {
            mergedLinkOriginalStart = p0
            mergedLinkOriginalEnd = p1
          }
        }
      }
      context.setDrawnObjects((prev: { objects: unknown[]; selectedIds: string[] }) => {
        const next = (prev.objects as Array<Record<string, unknown>>)
          .filter(obj => obj.id !== polygonId)
          .map(obj => {
            if (obj.id !== polygonIdA) return obj
            return {
              ...obj,
              points: mp.points,
              contained_links: contained,
              mergedLinkName: primary ?? undefined,
              mergedLinkOriginalStart: mergedLinkOriginalStart ?? obj.mergedLinkOriginalStart,
              mergedLinkOriginalEnd: mergedLinkOriginalEnd ?? obj.mergedLinkOriginalEnd,
              fillColor: objA.fillColor ?? obj.fillColor,
              strokeColor: objA.strokeColor ?? obj.strokeColor,
              fillOpacity: 0.25,
              contained_links_valid: true,
              ...(objA.z_level !== undefined && { z_level: objA.z_level })
            }
          })
        return { ...prev, objects: next, selectedIds: [] }
      })
      context.setMergePolygonState(initialMergePolygonState)
      context.setSelectedLinks([])
      context.showStatus(`✓ Merged polygons (${contained.length} link(s))`, 'success', 3000)
    })
    return true
  }
  return true
}

/** Internal: perform merge via backend and reset state. */
function completeMerge(context: ToolContext, polygonId: string): boolean {
  const polygon = context.drawnObjects.objects.find((obj: { id: string }) => obj.id === polygonId) as { id: string; name?: string; points: CanvasPoint[] } | undefined
  if (!polygon?.points || polygon.points.length < 3) {
    context.showStatus('Error: Could not find polygon or invalid points', 'error', 2000)
    context.setMergePolygonState(initialMergePolygonState)
    return false
  }
  if (!context.apiMergePolygon) {
    context.showStatus('Merge requires backend connection', 'error', 3000)
    context.setMergePolygonState(initialMergePolygonState)
    return false
  }
  context.apiMergePolygon({ polygonId, polygonPoints: polygon.points }).then(data => {
    if (data.status === 'success' && data.polygon?.contained_links?.length) {
      context.showStatus(`Merged polygon with ${data.polygon.contained_links.length} link(s)`, 'success', 3000)
    } else if (data.status === 'success') {
      context.showStatus('No fully bounded links inside polygon', 'warning', 3000)
    } else {
      context.showStatus(data.message ?? 'Merge failed', 'error', 3000)
    }
    context.setMergePolygonState(initialMergePolygonState)
    context.setSelectedLinks([])
  })
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

    const findClickedPolygon = (includeMerged = false) => {
      return context.drawnObjects.objects.find(obj => {
        if (obj.type !== 'polygon' || obj.points.length < 3) return false
        if (!includeMerged && (obj as { mergedLinkName?: string }).mergedLinkName) return false
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
        context.showStatus(`Selected link "${nearestLink.name}" — click a polygon form to merge with`, 'action')
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
        context.showStatus(`Selected polygon "${(clickedPolygon as { name?: string }).name ?? clickedPolygon.id}" — click a link or another polygon form to merge with`, 'action')
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
        context.showStatus('Select a link or a polygon form to begin merge', 'info', 2000)
      }
      return true
    }

    if (mergePolygonState.step === 'polygon_selected') {
      if (nearestLink) {
        const success = completeMerge(context, mergePolygonState.selectedPolygonId!)
        if (!success) {
          context.showStatus('Link endpoints must be inside the polygon. Try another link.', 'warning', 2500)
        }
        return true
      }

      const clickedPolygon = findClickedPolygon(true)
      if (clickedPolygon && clickedPolygon.id !== mergePolygonState.selectedPolygonId) {
        handleMergePolygonClick(context, {
          polygonId: clickedPolygon.id,
          polygonName: (clickedPolygon as { name?: string }).name ?? clickedPolygon.id,
          polygonPoints: clickedPolygon.points
        })
        return true
      }

      context.showStatus('Click a link or another polygon form to merge with', 'info', 2000)
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
        context.showStatus(`Switched to link "${nearestLink.name}" — click a polygon form to merge with`, 'action')
        return true
      }

      const clickedPolygon = findClickedPolygon()
      if (clickedPolygon) {
        const success = completeMerge(context, clickedPolygon.id)
        if (!success) {
          context.showStatus('Link endpoints must be inside the polygon. Try another polygon.', 'warning', 2500)
        }
        return true
      }

      context.showStatus('Click a polygon form to merge with the selected link', 'info', 2000)
      return true
    }

    return true
  }
}
