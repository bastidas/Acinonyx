/**
 * useToolContext
 *
 * Builds ToolContext for per-tool handlers (select, draw link, merge, etc.).
 * Single place for tool context construction; BuilderTab only wires inputs and passes result down.
 */

import { useMemo } from 'react'
import type { RefObject } from 'react'
import type { ToolContext } from '../toolHandlers/types'
import type { LinkageDocument, PylinkDocument } from '../types'
import { PIXELS_PER_UNIT } from '../constants'
import { JOINT_SNAP_THRESHOLD, MERGE_THRESHOLD } from '../../BuilderTools'
import type { DrawnObjectsState } from '../../BuilderTools'
import type { TargetPath } from '../../BuilderTools'

type BuildToolContextParams = Omit<
  ToolContext,
  | 'getPointFromEvent'
  | 'snapThreshold'
  | 'mergeThreshold'
  | 'getLinkMeta'
  | 'mergeLinkThreshold'
  | 'linksMeta'
  | 'setDrawnObjects'
  | 'setTargetPaths'
  | 'createDrawnObject'
  | 'createTargetPath'
> & {
  setLinkageDoc?: (value: LinkageDocument | ((prev: LinkageDocument) => LinkageDocument)) => void
} & {
  linkMeta: PylinkDocument['meta']['links']
  setDrawnObjects: React.Dispatch<React.SetStateAction<DrawnObjectsState>>
  setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
  createDrawnObject: (
    type: 'polygon',
    points: [number, number][],
    existingIds: string[]
  ) => { id: string; type: string; points: [number, number][]; name: string }
  createTargetPath: (points: [number, number][], existingPaths: TargetPath[]) => TargetPath
  viewport: { zoom: number; panX: number; panY: number }
}

function buildToolContext(params: BuildToolContextParams): ToolContext {
  const {
    canvasRef: cr,
    pixelsToUnits: pu,
    viewport: vp,
    linkMeta,
    setDrawnObjects: setDrawn,
    setTargetPaths: setPaths,
    createTargetPath: createPath,
    createDrawnObject: createDrawn,
    ...rest
  } = params
  return {
    ...rest,
    linksMeta: linkMeta,
    canvasRef: cr as RefObject<SVGSVGElement | HTMLDivElement | null>,
    pixelsToUnits: pu,
    getPointFromEvent: (event: React.MouseEvent<SVGSVGElement>) => {
      if (!cr.current) return null
      const rect = cr.current.getBoundingClientRect()
      const screenPx = event.clientX - rect.left
      const screenPy = event.clientY - rect.top
      const x = (screenPx - vp.panX) / vp.zoom / PIXELS_PER_UNIT
      const y = (screenPy - vp.panY) / vp.zoom / PIXELS_PER_UNIT
      return [x, y] as [number, number]
    },
    snapThreshold: JOINT_SNAP_THRESHOLD,
    mergeThreshold: MERGE_THRESHOLD,
    getLinkMeta: (linkName: string) => {
      const m = linkMeta[linkName]
      return m ? { connects: m.connects as [string, string], color: m.color } : null
    },
    mergeLinkThreshold: 8.0,
    setDrawnObjects: setDrawn as unknown as ToolContext['setDrawnObjects'],
    createDrawnObject: createDrawn as unknown as ToolContext['createDrawnObject'],
    setTargetPaths: setPaths as unknown as ToolContext['setTargetPaths'],
    createTargetPath: createPath as unknown as ToolContext['createTargetPath']
  } as ToolContext
}

export interface UseToolContextParams extends Omit<BuildToolContextParams, 'apiMergePolygon'> {
  /** Build linkage with current positions for merge-polygon API request. */
  buildLinkageWithCurrentPositions: (
    linkage: { nodes?: Record<string, { position?: number[] }>; edges?: unknown },
    getPos: (id: string) => [number, number] | null
  ) => { nodes: Record<string, { position?: number[] }>; edges?: unknown }
}

export function useToolContext(params: UseToolContextParams): ToolContext {
  const {
    buildLinkageWithCurrentPositions,
    linkageDoc,
    drawnObjects,
    getJointPosition,
    setDrawnObjects,
    setLinkageDoc,
    ...rest
  } = params

  return useMemo(() => {
    const apiMergePolygon = async (args: {
      polygonId: string
      polygonPoints: [number, number][]
      selectedLinkName?: string
      attachOnlyIfSingleContainedLink?: boolean
    }) => {
      const { polygonId, polygonPoints, selectedLinkName, attachOnlyIfSingleContainedLink } = args
      const linkageForRequest = buildLinkageWithCurrentPositions(linkageDoc.linkage, getJointPosition)
      const body = {
        pylink_data: { linkage: linkageForRequest, meta: linkageDoc.meta },
        polygon_id: polygonId,
        polygon_points: polygonPoints,
        ...(selectedLinkName != null && { selected_link_name: selectedLinkName })
      }
      const res = await fetch('/api/merge-polygon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = await res.json()
      if (data.status !== 'success' || !data.polygon) return data
      const p = data.polygon
      const selectedFullyInside = p.selected_link_fully_inside
      if (selectedLinkName != null && selectedFullyInside === false) {
        return data
      }
      const contained = p.contained_links as string[] | undefined
      if (
        attachOnlyIfSingleContainedLink &&
        contained &&
        contained.length > 1
      ) {
        return data
      }
      setDrawnObjects(prev => {
        const polygonObj = prev.objects.find((o: { id: string }) => o.id === polygonId) as
          | { z_level?: number }
          | undefined
        const firstLinkZ = p.contained_links?.length
          ? (linkageDoc.meta?.edges?.[p.contained_links[0]] as { zlevel?: number })?.zlevel
          : undefined
        const polygonZ = polygonObj?.z_level ?? firstLinkZ
        return {
          ...prev,
          objects: prev.objects.map(obj =>
            obj.id === polygonId
              ? {
                  ...obj,
                  contained_links: p.contained_links ?? [],
                  mergedLinkName: p.mergedLinkName ?? undefined,
                  mergedLinkOriginalStart: p.mergedLinkOriginalStart ?? undefined,
                  mergedLinkOriginalEnd: p.mergedLinkOriginalEnd ?? undefined,
                  fillColor: p.fill_color ?? obj.fillColor,
                  strokeColor: p.stroke_color ?? obj.strokeColor,
                  fillOpacity: 0.25,
                  contained_links_valid: true,
                  ...(polygonZ !== undefined && { z_level: polygonZ })
                }
              : obj
          )
        }
      })
      if (p.contained_links?.length && p.fill_color) {
        const polygonObj = drawnObjects.objects.find(
          (o: { id: string }) => o.id === polygonId
        ) as { z_level?: number } | undefined
        const firstLinkZ = (linkageDoc.meta?.edges?.[p.contained_links[0]] as { zlevel?: number })
          ?.zlevel
        const polygonZ = polygonObj?.z_level ?? firstLinkZ
        setLinkageDoc?.(prev => {
          const edges = { ...prev.meta.edges }
          for (const lid of p.contained_links) {
            const existing = edges[lid] ?? {}
            edges[lid] = {
              ...existing,
              color: p.fill_color,
              ...(polygonZ !== undefined && { zlevel: polygonZ })
            }
          }
          return { ...prev, meta: { ...prev.meta, edges } }
        })
      }
      return data
    }

    const apiMergeTwoPolygons = async (args: {
      polygonIdA: string
      polygonPointsA: [number, number][]
      polygonIdB: string
      polygonPointsB: [number, number][]
    }) => {
      const { polygonIdA, polygonPointsA, polygonIdB, polygonPointsB } = args
      const linkageForRequest = buildLinkageWithCurrentPositions(linkageDoc.linkage, getJointPosition)
      const body = {
        pylink_data: { linkage: linkageForRequest, meta: linkageDoc.meta },
        polygon_id_a: polygonIdA,
        polygon_points_a: polygonPointsA,
        polygon_id_b: polygonIdB,
        polygon_points_b: polygonPointsB
      }
      const res = await fetch('/api/merge-two-polygons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      return (await res.json()) as import('../toolHandlers/types').MergeTwoPolygonsApiResult
    }

    return buildToolContext({
      ...rest,
      linkageDoc,
      drawnObjects,
      getJointPosition,
      setDrawnObjects,
      setLinkageDoc,
      apiMergePolygon,
      apiMergeTwoPolygons
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- large params; same deps as original BuilderTab useMemo
  }, [
    buildLinkageWithCurrentPositions,
    linkageDoc,
    drawnObjects,
    getJointPosition,
    setDrawnObjects,
    setLinkageDoc,
    rest.canvasRef,
    rest.pixelsToUnits,
    rest.viewport,
    rest.toolMode,
    rest.setToolMode,
    rest.getJointsWithPositions,
    rest.getLinksWithPositions,
    rest.findNearestJoint,
    rest.findNearestLink,
    rest.findMergeTarget,
    rest.jointMergeRadius,
    rest.dragState,
    rest.setDragState,
    rest.selectedJoints,
    rest.setSelectedJoints,
    rest.selectedLinks,
    rest.setSelectedLinks,
    rest.moveJoint,
    rest.moveTwoJoints,
    rest.mergeJoints,
    rest.resetAnimationToFirstFrame,
    rest.pauseAnimation,
    rest.showStatus,
    rest.clearStatus,
    rest.linkCreationState,
    rest.setLinkCreationState,
    rest.setPreviewLine,
    rest.createLinkWithRevoluteDefault,
    rest.deleteJoint,
    rest.deleteLink,
    rest.handleDeleteSelected,
    rest.measureState,
    rest.setMeasureState,
    rest.setMeasurementMarkers,
    rest.calculateDistance,
    rest.groupSelectionState,
    rest.setGroupSelectionState,
    rest.findElementsInBox,
    rest.enterMoveGroupMode,
    rest.polygonDrawState,
    rest.setPolygonDrawState,
    rest.createDrawnObject,
    rest.mergePolygonState,
    rest.setMergePolygonState,
    rest.linkMeta,
    rest.isPointInPolygon,
    rest.areLinkEndpointsInPolygon,
    rest.getDefaultColor,
    rest.transformPolygonPoints,
    rest.pathDrawState,
    rest.setPathDrawState,
    rest.targetPaths,
    rest.setTargetPaths,
    rest.createTargetPath,
    rest.setSelectedPathId,
    rest.apiFindAssociatedPolygons,
    rest.exploreTrajectoriesState,
    rest.setExploreTrajectoriesState,
    rest.runExploreTrajectories,
    rest.runExploreTrajectoriesSecond,
    rest.runExploreTrajectoriesCombinatorial,
    rest.getNodeShowPath,
    rest.getJointType
  ])
}
