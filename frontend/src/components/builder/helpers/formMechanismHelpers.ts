/**
 * Forms vs mechanism selection, containment, and association validation.
 */

import type { LinkMetaData } from '../../BuilderTools'
import { areLinkEndpointsInPolygon, findConnectedMechanism } from '../../BuilderTools'

export type AxisAlignedBox = { minX: number; maxX: number; minY: number; maxY: number }

export function boxFromCorners(x1: number, y1: number, x2: number, y2: number): AxisAlignedBox {
  return {
    minX: Math.min(x1, x2),
    maxX: Math.max(x1, x2),
    minY: Math.min(y1, y2),
    maxY: Math.max(y1, y2)
  }
}

export function isPointInAxisAlignedBox(p: [number, number], box: AxisAlignedBox): boolean {
  return p[0] >= box.minX && p[0] <= box.maxX && p[1] >= box.minY && p[1] <= box.maxY
}

export function isAxisAlignedRectContainingPolygonVertices(
  points: [number, number][],
  box: AxisAlignedBox
): boolean {
  if (points.length < 3) return false
  return points.every(pt => isPointInAxisAlignedBox(pt, box))
}

export function areLinkEndpointsInBox(
  linkId: string,
  box: AxisAlignedBox,
  getJointPosition: (jointName: string) => [number, number] | null,
  linksMeta: Record<string, { connects: string[] }>
): boolean {
  const meta = linksMeta[linkId]
  if (!meta?.connects || meta.connects.length < 2) return false
  const a = getJointPosition(meta.connects[0]!)
  const b = getJointPosition(meta.connects[1]!)
  if (!a || !b) return false
  return isPointInAxisAlignedBox(a, box) && isPointInAxisAlignedBox(b, box)
}

export type FormForGroupSelect = {
  points: [number, number][]
  contained_links?: string[]
  mergedLinkName?: string
}

/**
 * Hybrid rule: general polygon = all vertices in box; single-link form = both link endpoints in box.
 */
export function isFormFullyInsideGroupSelectBox(
  obj: FormForGroupSelect,
  box: AxisAlignedBox,
  getJointPosition: (jointName: string) => [number, number] | null,
  linksMeta: Record<string, { connects: string[] }>
): boolean {
  const cl = obj.contained_links ?? []
  if (cl.length === 1) {
    return areLinkEndpointsInBox(cl[0]!, box, getJointPosition, linksMeta)
  }
  if (cl.length === 0 && obj.mergedLinkName) {
    return areLinkEndpointsInBox(obj.mergedLinkName, box, getJointPosition, linksMeta)
  }
  return isAxisAlignedRectContainingPolygonVertices(obj.points, box)
}

/** Document node positions only (for post-drag association checks). */
export function getJointPositionFromLinkageDoc(
  linkageDoc: { linkage?: { nodes?: Record<string, { position?: number[] }> } } | null,
  jointName: string
): [number, number] | null {
  const pos = linkageDoc?.linkage?.nodes?.[jointName]?.position
  if (!Array.isArray(pos) || pos.length < 2) return null
  return [Number(pos[0]), Number(pos[1])]
}

function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false
  for (const x of a) {
    if (!b.has(x)) return false
  }
  return true
}

/**
 * True when selected joints+links are exactly one connected component (full mechanism in the box).
 */
export function selectionIsExactlyOneMechanism(
  selectedJointIds: string[],
  selectedLinkIds: string[],
  links: Record<string, LinkMetaData>
): boolean {
  const jSet = new Set(selectedJointIds)
  const lSet = new Set(selectedLinkIds)
  if (jSet.size === 0 && lSet.size === 0) return false

  let root: string | null = null
  for (const j of jSet) {
    root = j
    break
  }
  if (root == null) {
    for (const lid of lSet) {
      const m = links[lid]
      if (m?.connects?.[0]) {
        root = m.connects[0]!
        break
      }
    }
  }
  if (root == null) return false

  const mech = findConnectedMechanism(root, links)
  return setsEqual(new Set(mech.joints), jSet) && setsEqual(new Set(mech.links), lSet)
}

/** Forms tied to any link in the mechanism (merge primary or contained overlap). */
export function polygonIdsTouchingMechanismLinks(
  objects: Array<{
    id: string
    type?: string
    mergedLinkName?: string
    contained_links?: string[]
  }>,
  mechanismLinkIds: string[]
): string[] {
  const linkSet = new Set(mechanismLinkIds)
  return objects
    .filter(o => o.type === 'polygon' || o.type == null)
    .filter(
      o =>
        (o.mergedLinkName != null && linkSet.has(o.mergedLinkName)) ||
        (o.contained_links ?? []).some(lid => linkSet.has(lid))
    )
    .map(o => o.id)
}

export const DISSOCIATED_FORM_FILL = '#bdbdbd'
export const DISSOCIATED_FORM_STROKE = '#757575'

export type PolygonFormForDissociate = {
  contained_links?: string[]
  mergedLinkName?: string
  mergedLinkOriginalStart?: [number, number]
  mergedLinkOriginalEnd?: [number, number]
  contained_links_valid?: boolean
  fillColor?: string
  strokeColor?: string
}

/** Clear link association and grey out (decorative / disconnected form). */
export function dissociateFormFields<T extends PolygonFormForDissociate>(obj: T): T {
  return {
    ...obj,
    contained_links: [],
    mergedLinkName: undefined,
    mergedLinkOriginalStart: undefined,
    mergedLinkOriginalEnd: undefined,
    contained_links_valid: undefined,
    fillColor: DISSOCIATED_FORM_FILL,
    strokeColor: DISSOCIATED_FORM_STROKE
  }
}

function linkIdsForForm(obj: { contained_links?: string[]; mergedLinkName?: string }): string[] {
  const s = new Set<string>()
  for (const lid of obj.contained_links ?? []) s.add(lid)
  if (obj.mergedLinkName) s.add(obj.mergedLinkName)
  return [...s]
}

/**
 * If any associated link's endpoints are not both inside the polygon (world `points`), dissociate the form.
 */
export function validatePolygonFormAssociations<T extends PolygonFormForDissociate & { points: [number, number][]; type?: string }>(
  objects: T[],
  getJointPosition: (jointName: string) => [number, number] | null,
  linksMeta: Record<string, { connects: string[] }>
): T[] {
  return objects.map(obj => {
    if (obj.type != null && obj.type !== 'polygon') return obj
    const lids = linkIdsForForm(obj)
    if (lids.length === 0) return obj
    const poly = obj.points
    if (poly.length < 3) return obj

    for (const lid of lids) {
      const meta = linksMeta[lid]
      if (!meta?.connects || meta.connects.length < 2) {
        return dissociateFormFields(obj)
      }
      const p0 = getJointPosition(meta.connects[0]!)
      const p1 = getJointPosition(meta.connects[1]!)
      if (!p0 || !p1 || !areLinkEndpointsInPolygon(p0, p1, poly)) {
        return dissociateFormFields(obj)
      }
    }
    return obj
  })
}
