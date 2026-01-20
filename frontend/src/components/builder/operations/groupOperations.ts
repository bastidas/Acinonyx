/**
 * Group Operations
 *
 * Pure functions for group selection, movement, and merge operations.
 * These functions take the current document state and return the new state.
 */

import {
  PylinkDocument,
  GetJointPositionFn
} from './types'
import { deleteJoint, translateGroupRigid } from './jointOperations'
import { deleteLink } from './linkOperations'

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH DELETE ITEMS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Result of a batch deletion operation
 */
export interface BatchDeletionResult {
  newDoc: PylinkDocument
  deletedJoints: string[]
  deletedLinks: string[]
  message: string
}

/**
 * Delete multiple joints and links in a single operation
 *
 * Order matters:
 * 1. Delete links first (to avoid orphan issues)
 * 2. Then delete joints
 *
 * @param doc - Current document state
 * @param jointNames - Joints to delete
 * @param linkNames - Links to delete
 * @returns Result with new document and deletion details
 */
export function batchDeleteItems(
  doc: PylinkDocument,
  jointNames: string[],
  linkNames: string[]
): BatchDeletionResult {
  let currentDoc = doc
  const allDeletedJoints: string[] = []
  const allDeletedLinks: string[] = []

  // Delete links first
  for (const linkName of linkNames) {
    if (currentDoc.meta.links[linkName]) {
      const result = deleteLink(currentDoc, linkName)
      currentDoc = result.newDoc
      allDeletedLinks.push(linkName)
      // Track any joints that became orphans
      allDeletedJoints.push(...result.orphanedJoints)
    }
  }

  // Then delete joints (some may have been deleted as orphans already)
  for (const jointName of jointNames) {
    const jointExists = currentDoc.pylinkage.joints.some(j => j.name === jointName)
    if (jointExists && !allDeletedJoints.includes(jointName)) {
      const result = deleteJoint(currentDoc, jointName)
      currentDoc = result.newDoc
      allDeletedJoints.push(...result.deletedJoints)
      allDeletedLinks.push(...result.deletedLinks.filter(l => !allDeletedLinks.includes(l)))
    }
  }

  // Build message
  const parts: string[] = []
  if (allDeletedJoints.length > 0) {
    parts.push(`${allDeletedJoints.length} joint(s)`)
  }
  if (allDeletedLinks.length > 0) {
    parts.push(`${allDeletedLinks.length} link(s)`)
  }

  return {
    newDoc: currentDoc,
    deletedJoints: allDeletedJoints,
    deletedLinks: allDeletedLinks,
    message: `Deleted ${parts.join(' + ')}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP POSITION CAPTURE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Capture the current positions of a group of joints
 *
 * @param jointNames - Names of joints to capture
 * @param getJointPosition - Position lookup function
 * @returns Map of joint names to positions
 */
export function captureGroupPositions(
  jointNames: string[],
  getJointPosition: GetJointPositionFn
): Record<string, [number, number]> {
  const positions: Record<string, [number, number]> = {}

  for (const jointName of jointNames) {
    const pos = getJointPosition(jointName)
    if (pos) {
      positions[jointName] = pos
    }
  }

  return positions
}

/**
 * Calculate the bounding box of a group of joints
 *
 * @param positions - Map of joint names to positions
 * @returns Bounding box { minX, minY, maxX, maxY } or null if empty
 */
export function calculateGroupBounds(
  positions: Record<string, [number, number]>
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const posArray = Object.values(positions)
  if (posArray.length === 0) return null

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const [x, y] of posArray) {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
  }

  return { minX, minY, maxX, maxY }
}

/**
 * Calculate the center of a group of joints
 *
 * @param positions - Map of joint names to positions
 * @returns Center position [x, y] or null if empty
 */
export function calculateGroupCenter(
  positions: Record<string, [number, number]>
): [number, number] | null {
  const posArray = Object.values(positions)
  if (posArray.length === 0) return null

  let sumX = 0
  let sumY = 0

  for (const [x, y] of posArray) {
    sumX += x
    sumY += y
  }

  return [sumX / posArray.length, sumY / posArray.length]
}

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP MOVEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Move a group of joints by a delta amount (rigid translation)
 *
 * This is a convenience wrapper around translateGroupRigid that
 * captures original positions and applies the delta.
 *
 * @param doc - Current document state
 * @param jointNames - Joints to move
 * @param originalPositions - Positions at start of drag
 * @param dx - Delta X
 * @param dy - Delta Y
 * @returns New document state
 */
export function moveGroup(
  doc: PylinkDocument,
  jointNames: string[],
  originalPositions: Record<string, [number, number]>,
  dx: number,
  dy: number
): PylinkDocument {
  const result = translateGroupRigid(doc, jointNames, originalPositions, dx, dy)
  return result.newDoc
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONNECTED MECHANISM FINDING
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find all joints connected to a starting joint via links
 * (breadth-first traversal of the linkage graph)
 *
 * @param startJoint - Starting joint name
 * @param links - Link definitions
 * @returns Array of connected joint names (including startJoint)
 */
export function findConnectedJoints(
  startJoint: string,
  links: Record<string, { connects: string[] }>
): string[] {
  const connected = new Set<string>([startJoint])
  const queue = [startJoint]

  while (queue.length > 0) {
    const current = queue.shift()!

    // Find all links that include this joint
    for (const link of Object.values(links)) {
      if (link.connects.includes(current)) {
        for (const joint of link.connects) {
          if (!connected.has(joint)) {
            connected.add(joint)
            queue.push(joint)
          }
        }
      }
    }
  }

  return Array.from(connected)
}

/**
 * Find all links that connect joints in a given set
 *
 * @param jointNames - Set of joint names
 * @param links - Link definitions
 * @returns Array of link names where both endpoints are in the set
 */
export function findLinksInGroup(
  jointNames: string[],
  links: Record<string, { connects: string[] }>
): string[] {
  const jointSet = new Set(jointNames)
  const result: string[] = []

  for (const [linkName, link] of Object.entries(links)) {
    if (link.connects.every(j => jointSet.has(j))) {
      result.push(linkName)
    }
  }

  return result
}

/**
 * Find all links that have at least one endpoint in a given set
 *
 * @param jointNames - Set of joint names
 * @param links - Link definitions
 * @returns Array of link names where at least one endpoint is in the set
 */
export function findLinksConnectedToGroup(
  jointNames: string[],
  links: Record<string, { connects: string[] }>
): string[] {
  const jointSet = new Set(jointNames)
  const result: string[] = []

  for (const [linkName, link] of Object.entries(links)) {
    if (link.connects.some(j => jointSet.has(j))) {
      result.push(linkName)
    }
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find joints within a rectangular selection box
 *
 * @param joints - Array of joint data with names
 * @param getJointPosition - Position lookup function
 * @param boxStart - Start corner of selection box
 * @param boxEnd - End corner of selection box
 * @returns Array of joint names within the box
 */
export function findJointsInBox(
  jointNames: string[],
  getJointPosition: GetJointPositionFn,
  boxStart: [number, number],
  boxEnd: [number, number]
): string[] {
  const minX = Math.min(boxStart[0], boxEnd[0])
  const maxX = Math.max(boxStart[0], boxEnd[0])
  const minY = Math.min(boxStart[1], boxEnd[1])
  const maxY = Math.max(boxStart[1], boxEnd[1])

  const result: string[] = []

  for (const jointName of jointNames) {
    const pos = getJointPosition(jointName)
    if (pos) {
      const [x, y] = pos
      if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
        result.push(jointName)
      }
    }
  }

  return result
}

/**
 * Find links where both endpoints are within a selection box
 *
 * @param links - Link definitions
 * @param getJointPosition - Position lookup function
 * @param boxStart - Start corner of selection box
 * @param boxEnd - End corner of selection box
 * @returns Array of link names within the box
 */
export function findLinksInBox(
  links: Record<string, { connects: string[] }>,
  getJointPosition: GetJointPositionFn,
  boxStart: [number, number],
  boxEnd: [number, number]
): string[] {
  const minX = Math.min(boxStart[0], boxEnd[0])
  const maxX = Math.max(boxStart[0], boxEnd[0])
  const minY = Math.min(boxStart[1], boxEnd[1])
  const maxY = Math.max(boxStart[1], boxEnd[1])

  const isInBox = (pos: [number, number]) => {
    const [x, y] = pos
    return x >= minX && x <= maxX && y >= minY && y <= maxY
  }

  const result: string[] = []

  for (const [linkName, link] of Object.entries(links)) {
    const pos0 = getJointPosition(link.connects[0])
    const pos1 = getJointPosition(link.connects[1])

    if (pos0 && pos1 && isInBox(pos0) && isInBox(pos1)) {
      result.push(linkName)
    }
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// DUPLICATE GROUP
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Result of a group duplication operation
 */
export interface DuplicateGroupResult {
  newDoc: PylinkDocument
  newJointNames: Record<string, string>  // old name -> new name
  newLinkNames: Record<string, string>   // old name -> new name
}

/**
 * Duplicate a group of joints and their connecting links
 *
 * @param doc - Current document state
 * @param jointNames - Joints to duplicate
 * @param offset - Position offset for the duplicates [dx, dy]
 * @param getJointPosition - Position lookup function
 * @returns Result with new document and name mappings
 */
export function duplicateGroup(
  doc: PylinkDocument,
  jointNames: string[],
  offset: [number, number],
  getJointPosition: GetJointPositionFn
): DuplicateGroupResult {
  const jointNameMap: Record<string, string> = {}
  const linkNameMap: Record<string, string> = {}

  let newJoints = [...doc.pylinkage.joints]
  let newMetaJoints = { ...doc.meta.joints }
  let newMetaLinks = { ...doc.meta.links }
  let newSolveOrder = [...doc.pylinkage.solve_order]

  const existingJointNames = new Set(doc.pylinkage.joints.map(j => j.name))
  const existingLinkNames = new Set(Object.keys(doc.meta.links))

  // Generate unique name
  const generateUniqueName = (baseName: string, existingSet: Set<string>): string => {
    let counter = 1
    let newName = `${baseName}_copy`
    while (existingSet.has(newName)) {
      newName = `${baseName}_copy${counter}`
      counter++
    }
    return newName
  }

  // Duplicate joints
  for (const jointName of jointNames) {
    const joint = doc.pylinkage.joints.find(j => j.name === jointName)
    if (!joint) continue

    const newName = generateUniqueName(jointName, existingJointNames)
    jointNameMap[jointName] = newName
    existingJointNames.add(newName)

    const pos = getJointPosition(jointName)
    if (!pos) continue

    const newX = pos[0] + offset[0]
    const newY = pos[1] + offset[1]

    if (joint.type === 'Static') {
      newJoints.push({
        type: 'Static',
        name: newName,
        x: newX,
        y: newY
      })
    } else {
      // For non-static joints, we'll update references after all joints are created
      newJoints.push({ ...joint, name: newName })
      newMetaJoints[newName] = {
        ...newMetaJoints[jointName],
        x: newX,
        y: newY
      }
    }

    newSolveOrder.push(newName)
  }

  // Update references in duplicated non-static joints
  newJoints = newJoints.map(joint => {
    if (!jointNameMap[joint.name] || joint.name === jointNameMap[joint.name]) {
      // Not a duplicated joint, skip
      return joint
    }

    // This is one of our duplicated joints - update its references
    if (joint.type === 'Crank' && jointNameMap[joint.joint0.ref]) {
      return { ...joint, joint0: { ref: jointNameMap[joint.joint0.ref] } }
    }
    if (joint.type === 'Revolute') {
      const updated = { ...joint }
      if (jointNameMap[joint.joint0.ref]) {
        updated.joint0 = { ref: jointNameMap[joint.joint0.ref] }
      }
      if (jointNameMap[joint.joint1.ref]) {
        updated.joint1 = { ref: jointNameMap[joint.joint1.ref] }
      }
      return updated
    }
    return joint
  })

  // Duplicate links between the duplicated joints
  const links = findLinksInGroup(jointNames, doc.meta.links)
  for (const linkName of links) {
    const link = doc.meta.links[linkName]
    const newName = generateUniqueName(linkName, existingLinkNames)
    linkNameMap[linkName] = newName
    existingLinkNames.add(newName)

    newMetaLinks[newName] = {
      ...link,
      connects: link.connects.map(j => jointNameMap[j] || j)
    }
  }

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newMetaJoints,
        links: newMetaLinks
      }
    },
    newJointNames: jointNameMap,
    newLinkNames: linkNameMap
  }
}
