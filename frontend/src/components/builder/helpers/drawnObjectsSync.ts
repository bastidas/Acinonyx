/**
 * Keep drawn canvas objects' edge id references aligned with linkage.edges keys
 * after rename/delete (source-of-truth consistency for save/export).
 */

function objectTouchesDeletedEdge(obj: unknown, deleted: ReadonlySet<string>): boolean {
  if (!obj || typeof obj !== 'object') return false
  const o = obj as Record<string, unknown>
  const merged = o.mergedLinkName
  if (typeof merged === 'string' && deleted.has(merged)) return true
  const links = o.contained_links
  if (Array.isArray(links)) {
    for (const id of links) {
      if (typeof id === 'string' && deleted.has(id)) return true
    }
  }
  return false
}

/**
 * Remove drawn objects that reference any deleted edge id via mergedLinkName or contained_links.
 */
export function removeDrawnObjectsReferencingDeletedEdges(
  objects: unknown[],
  deletedEdgeIds: ReadonlySet<string>
): { objects: unknown[]; removedIds: string[] } {
  if (deletedEdgeIds.size === 0) {
    return { objects: objects.slice(), removedIds: [] }
  }
  const removedIds: string[] = []
  const kept: unknown[] = []
  for (const obj of objects) {
    if (objectTouchesDeletedEdge(obj, deletedEdgeIds)) {
      const id = (obj as { id?: unknown }).id
      if (typeof id === 'string') removedIds.push(id)
    } else {
      kept.push(obj)
    }
  }
  return { objects: kept, removedIds }
}

/**
 * Rewrite edge id strings on polygons after an edge rename (object keys in linkage.edges).
 */
export function remapEdgeReferencesInDrawnObjects(
  objects: unknown[],
  oldEdgeId: string,
  newEdgeId: string
): unknown[] {
  if (oldEdgeId === newEdgeId) return objects.slice()
  return objects.map(obj => {
    if (!obj || typeof obj !== 'object') return obj
    const o = obj as Record<string, unknown>
    let next: Record<string, unknown> | null = null
    const ensureNext = () => {
      if (!next) next = { ...o }
      return next
    }
    if (o.mergedLinkName === oldEdgeId) {
      ensureNext().mergedLinkName = newEdgeId
    }
    if (Array.isArray(o.contained_links)) {
      const links = o.contained_links.filter((x): x is string => typeof x === 'string')
      if (links.some(id => id === oldEdgeId)) {
        ensureNext().contained_links = links.map(id => (id === oldEdgeId ? newEdgeId : id))
      }
    }
    return next ?? obj
  })
}
