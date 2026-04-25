/**
 * Apply a loaded LinkageDocument to builder state.
 *
 * Call this after validating format (e.g. isHypergraphFormat). The caller is
 * responsible for showStatus and any doc patching (e.g. target_joint for demos).
 */

import type { LinkageDocument } from '../types'

function makeUniqueId(baseId: string, taken: Set<string>): string {
  const base = baseId && baseId.trim() ? baseId : 'item'
  if (!taken.has(base)) {
    taken.add(base)
    return base
  }
  let idx = 1
  while (taken.has(`${base}_${idx}`)) idx += 1
  const next = `${base}_${idx}`
  taken.add(next)
  return next
}

function mergeDocuments(current: LinkageDocument, incoming: LinkageDocument): {
  doc: LinkageDocument
  nodeIdMap: Record<string, string>
  edgeIdMap: Record<string, string>
} {
  const mergedNodes = { ...current.linkage.nodes }
  const mergedEdges = { ...current.linkage.edges }
  const mergedHyperedges = { ...current.linkage.hyperedges }
  const mergedNodeMeta = { ...current.meta.nodes }
  const mergedEdgeMeta = { ...current.meta.edges }

  const takenNodeIds = new Set(Object.keys(mergedNodes))
  const takenEdgeIds = new Set(Object.keys(mergedEdges))
  const takenHyperedgeIds = new Set(Object.keys(mergedHyperedges))
  const nodeIdMap: Record<string, string> = {}
  const edgeIdMap: Record<string, string> = {}

  for (const [oldNodeId, node] of Object.entries(incoming.linkage.nodes)) {
    const nextNodeId = makeUniqueId(oldNodeId, takenNodeIds)
    nodeIdMap[oldNodeId] = nextNodeId
    mergedNodes[nextNodeId] = {
      ...node,
      id: nextNodeId,
      name: node.name === oldNodeId ? nextNodeId : node.name
    }
    const nodeMeta = incoming.meta.nodes[oldNodeId]
    if (nodeMeta) {
      mergedNodeMeta[nextNodeId] = { ...nodeMeta }
    }
  }

  for (const [oldEdgeId, edge] of Object.entries(incoming.linkage.edges)) {
    const nextEdgeId = makeUniqueId(oldEdgeId, takenEdgeIds)
    edgeIdMap[oldEdgeId] = nextEdgeId
    const source = nodeIdMap[edge.source] ?? edge.source
    const target = nodeIdMap[edge.target] ?? edge.target
    mergedEdges[nextEdgeId] = {
      ...edge,
      id: nextEdgeId,
      source,
      target
    }
    const edgeMeta = incoming.meta.edges[oldEdgeId]
    if (edgeMeta) {
      const metaWithConnects = edgeMeta as Record<string, unknown>
      const rawConnects = Array.isArray(metaWithConnects.connects) ? metaWithConnects.connects : undefined
      mergedEdgeMeta[nextEdgeId] = {
        ...edgeMeta,
        ...(rawConnects && rawConnects.length >= 2
          ? {
              connects: [
                nodeIdMap[String(rawConnects[0])] ?? String(rawConnects[0]),
                nodeIdMap[String(rawConnects[1])] ?? String(rawConnects[1])
              ]
            }
          : {})
      }
    }
  }

  for (const [oldHyperedgeId, hyperedge] of Object.entries(incoming.linkage.hyperedges)) {
    const nextHyperedgeId = makeUniqueId(oldHyperedgeId, takenHyperedgeIds)
    const remappedNodes = hyperedge.nodes.map(nodeId => nodeIdMap[nodeId] ?? nodeId)
    const remappedConstraints: Record<string, number> = {}
    for (const [pair, distance] of Object.entries(hyperedge.constraints)) {
      const [a, b] = pair.split(':')
      const ra = nodeIdMap[a] ?? a
      const rb = nodeIdMap[b] ?? b
      remappedConstraints[`${ra}:${rb}`] = distance
    }
    mergedHyperedges[nextHyperedgeId] = {
      ...hyperedge,
      id: nextHyperedgeId,
      nodes: remappedNodes,
      constraints: remappedConstraints
    }
  }

  return {
    doc: {
      ...current,
      linkage: {
        ...current.linkage,
        nodes: mergedNodes,
        edges: mergedEdges,
        hyperedges: mergedHyperedges
      },
      meta: {
        ...current.meta,
        nodes: mergedNodeMeta,
        edges: mergedEdgeMeta,
        target_joint:
          current.meta.target_joint ??
          (incoming.meta.target_joint ? nodeIdMap[incoming.meta.target_joint] ?? incoming.meta.target_joint : undefined)
      }
    },
    nodeIdMap,
    edgeIdMap
  }
}

export interface ApplyLoadedDocumentParams {
  doc: LinkageDocument
  setLinkageDoc: React.Dispatch<React.SetStateAction<LinkageDocument>>
  setDrawnObjects: React.Dispatch<React.SetStateAction<{ objects: unknown[]; selectedIds: string[] }>>
  setCanvases?: React.Dispatch<React.SetStateAction<LinkageDocument['canvases']>>
  setSelectedJoints: (ids: string[]) => void
  setSelectedLinks: (ids: string[]) => void
  clearTrajectory: () => void
  triggerMechanismChange: () => void
  /** When set, called before applying doc (e.g. clear link-creation undo stacks). */
  clearLinkageUndoStacks?: () => void
  /**
   * When true (default), replace the current mechanism + drawing with the loaded document.
   * When false, import/append the loaded mechanism + drawing into the current one without clearing.
   */
  clearDrawingOnLoad?: boolean
}

/**
 * Applies a loaded document to builder state: either replaces the current document
 * (`clearDrawingOnLoad` true) or appends the loaded mechanism + drawing without
 * clearing (`clearDrawingOnLoad` false), then clears joint/link selection, trajectory,
 * and triggers
 * mechanism re-run. Does not call showStatus; caller handles success/error messages.
 */
export function applyLoadedDocument(params: ApplyLoadedDocumentParams): void {
  const {
    doc,
    setLinkageDoc,
    setDrawnObjects,
    setCanvases,
    setSelectedJoints,
    setSelectedLinks,
    clearTrajectory,
    triggerMechanismChange,
    clearLinkageUndoStacks,
    clearDrawingOnLoad
  } = params

  const replaceDrawing = clearDrawingOnLoad !== false

  clearLinkageUndoStacks?.()

  if (replaceDrawing) {
    setLinkageDoc(doc)
    setDrawnObjects({
      objects: Array.isArray(doc.drawnObjects) ? doc.drawnObjects : [],
      selectedIds: []
    })
    if (setCanvases) {
      setCanvases(Array.isArray(doc.canvases) ? doc.canvases : [])
    }
  } else {
    const idMapsRef: { nodeIdMap: Record<string, string>; edgeIdMap: Record<string, string> } = {
      nodeIdMap: {},
      edgeIdMap: {}
    }
    setLinkageDoc(prev => {
      const merged = mergeDocuments(prev, doc)
      idMapsRef.nodeIdMap = merged.nodeIdMap
      idMapsRef.edgeIdMap = merged.edgeIdMap
      return merged.doc
    })
    setDrawnObjects(prev => {
      const existingIds = new Set(
        prev.objects
          .map(o => (o as { id?: string }).id)
          .filter((id): id is string => typeof id === 'string')
      )
      const incomingObjects = Array.isArray(doc.drawnObjects) ? doc.drawnObjects : []
      const remappedObjects = incomingObjects.map(obj => {
        const oldId = typeof obj.id === 'string' ? obj.id : 'object'
        const nextId = makeUniqueId(oldId, existingIds)
        const containedLinks = Array.isArray(obj.contained_links)
          ? obj.contained_links.map(linkId => idMapsRef.edgeIdMap[linkId] ?? linkId)
          : obj.contained_links
        const mergedLinkName = obj.mergedLinkName
          ? idMapsRef.edgeIdMap[obj.mergedLinkName] ?? obj.mergedLinkName
          : obj.mergedLinkName
        return {
          ...obj,
          id: nextId,
          contained_links: containedLinks,
          mergedLinkName
        }
      })
      return {
        ...prev,
        objects: [...prev.objects, ...remappedObjects]
      }
    })
    if (setCanvases) {
      setCanvases(prev => {
        const existingIds = new Set(
          (prev ?? [])
            .map(c => c?.id)
            .filter((id): id is string => typeof id === 'string')
        )
        const incoming = Array.isArray(doc.canvases) ? doc.canvases : []
        const remapped = incoming.map(canvas => ({
          ...canvas,
          id: makeUniqueId(canvas.id ?? 'canvas', existingIds)
        }))
        return [...(prev ?? []), ...remapped]
      })
    }
  }
  setSelectedJoints([])
  setSelectedLinks([])
  clearTrajectory()
  triggerMechanismChange()
}
