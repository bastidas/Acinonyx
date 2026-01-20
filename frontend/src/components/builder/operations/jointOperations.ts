/**
 * Joint Operations
 *
 * Pure functions for creating, updating, deleting, and manipulating joints.
 * These functions take the current document state and return the new state.
 */

import {
  PylinkDocument,
  StaticJoint,
  CrankJoint,
  RevoluteJoint,
  JointDeletionResult,
  JointMoveResult,
  GroupTranslateResult,
  JointMergeResult,
  RenameResult,
  UpdatePropertyResult,
  GetJointPositionFn,
  CalculateDistanceFn
} from './types'
import {
  calculateJointDeletionResult,
  calculateMergeResult
} from '../../BuilderTools'

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE JOINT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Delete a joint and all connected links, plus any resulting orphan joints
 *
 * @param doc - Current document state
 * @param jointName - Name of joint to delete
 * @returns Result with new document and deletion details
 */
export function deleteJoint(
  doc: PylinkDocument,
  jointName: string
): JointDeletionResult {
  const result = calculateJointDeletionResult(jointName, doc.meta.links)

  // Create new state - filter out all joints to be deleted (including orphans)
  const newLinks = { ...doc.meta.links }
  const newJoints = { ...doc.meta.joints }
  const newPylinkageJoints = doc.pylinkage.joints.filter(
    j => !result.jointsToDelete.includes(j.name)
  )
  const newSolveOrder = doc.pylinkage.solve_order.filter(
    name => !result.jointsToDelete.includes(name)
  )

  // Remove all connected links
  result.linksToDelete.forEach(linkName => {
    delete newLinks[linkName]
  })

  // Remove all joint metadata (including orphans)
  result.jointsToDelete.forEach(jName => {
    delete newJoints[jName]
  })

  const newDoc: PylinkDocument = {
    ...doc,
    pylinkage: {
      ...doc.pylinkage,
      joints: newPylinkageJoints,
      solve_order: newSolveOrder
    },
    meta: {
      joints: newJoints,
      links: newLinks
    }
  }

  // Build status message
  const orphanCount = result.jointsToDelete.length - 1
  const parts: string[] = [`Deleted ${jointName}`]
  if (result.linksToDelete.length > 0) {
    parts.push(`${result.linksToDelete.length} link(s)`)
  }
  if (orphanCount > 0) {
    parts.push(`${orphanCount} orphan(s)`)
  }

  return {
    newDoc,
    deletedJoints: result.jointsToDelete,
    deletedLinks: result.linksToDelete,
    message: parts.join(' + ')
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOVE JOINT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Move a joint to a new position
 *
 * For Static joints: updates position directly in pylinkage data
 * For Crank joints: preserves type and rotation, updates distance
 * For Revolute joints: preserves type, updates distances
 *
 * @param doc - Current document state
 * @param jointName - Name of joint to move
 * @param newPosition - New position [x, y]
 * @param getJointPosition - Function to get joint positions (for distance calculation)
 * @param calculateDistance - Function to calculate distance between points
 * @returns Result with new document
 */
export function moveJoint(
  doc: PylinkDocument,
  jointName: string,
  newPosition: [number, number],
  getJointPosition: GetJointPositionFn,
  calculateDistance: CalculateDistanceFn
): JointMoveResult | null {
  const jointIndex = doc.pylinkage.joints.findIndex(j => j.name === jointName)
  if (jointIndex === -1) return null

  const currentJoint = doc.pylinkage.joints[jointIndex]
  const newJoints = [...doc.pylinkage.joints]
  const newMetaJoints = { ...doc.meta.joints }

  if (currentJoint.type === 'Static') {
    // Static joints: update position directly in pylinkage data
    newJoints[jointIndex] = {
      type: 'Static',
      name: jointName,
      x: newPosition[0],
      y: newPosition[1]
    } as StaticJoint
    // Clear meta position since Static uses pylinkage x,y
    if (newMetaJoints[jointName]) {
      newMetaJoints[jointName] = { ...newMetaJoints[jointName], x: undefined, y: undefined }
    }
  } else if (currentJoint.type === 'Crank') {
    // Crank joints: preserve type and rotation speed, update distance to match new position
    const parentName = currentJoint.joint0.ref
    const parentPos = getJointPosition(parentName)
    if (parentPos) {
      const distance = calculateDistance(parentPos, newPosition)
      newJoints[jointIndex] = {
        type: 'Crank',
        name: jointName,
        joint0: currentJoint.joint0,
        distance: distance,
        angle: currentJoint.angle  // Keep original rotation speed
      } as CrankJoint
      // Update meta position for UI rendering
      newMetaJoints[jointName] = {
        ...newMetaJoints[jointName] || { color: '', zlevel: 0 },
        x: newPosition[0],
        y: newPosition[1]
      }
    }
  } else if (currentJoint.type === 'Revolute') {
    // Revolute joints: preserve type, update distances to match new position
    const parent0Name = currentJoint.joint0.ref
    const parent1Name = currentJoint.joint1.ref
    const parent0Pos = getJointPosition(parent0Name)
    const parent1Pos = getJointPosition(parent1Name)
    if (parent0Pos && parent1Pos) {
      const distance0 = calculateDistance(parent0Pos, newPosition)
      const distance1 = calculateDistance(parent1Pos, newPosition)
      newJoints[jointIndex] = {
        type: 'Revolute',
        name: jointName,
        joint0: currentJoint.joint0,
        joint1: currentJoint.joint1,
        distance0: distance0,
        distance1: distance1
      } as RevoluteJoint
      // Update meta position for UI rendering
      newMetaJoints[jointName] = {
        ...newMetaJoints[jointName] || { color: '', zlevel: 0 },
        x: newPosition[0],
        y: newPosition[1]
      }
    }
  }

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newJoints
      },
      meta: {
        ...doc.meta,
        joints: newMetaJoints
      }
    },
    movedJoint: jointName
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSLATE GROUP RIGID
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Move a group of joints to new positions based on original positions + delta.
 *
 * IMPORTANT: This does NOT recalculate distances or angles - it preserves the
 * exact structure of the mechanism and only applies a uniform translation.
 *
 * For Static joints:   set x, y to targetPosition (directly in pylinkage data)
 * For Crank joints:    set meta x, y to targetPosition (preserves distance and angle)
 * For Revolute joints: set meta x, y to targetPosition (preserves distance0 and distance1)
 *
 * @param doc - Current document state
 * @param jointNames - Array of joint names to move
 * @param originalPositions - Map of joint names to their original positions before drag started
 * @param dx - Delta X from drag start
 * @param dy - Delta Y from drag start
 * @returns Result with new document
 */
export function translateGroupRigid(
  doc: PylinkDocument,
  jointNames: string[],
  originalPositions: Record<string, [number, number]>,
  dx: number,
  dy: number
): GroupTranslateResult {
  if (jointNames.length === 0) {
    return { newDoc: doc, movedJoints: [] }
  }

  const newJoints = [...doc.pylinkage.joints]
  const newMetaJoints = { ...doc.meta.joints }

  for (const jointName of jointNames) {
    const originalPos = originalPositions[jointName]
    if (!originalPos) continue

    const targetX = originalPos[0] + dx
    const targetY = originalPos[1] + dy

    const jointIndex = newJoints.findIndex(j => j.name === jointName)
    if (jointIndex === -1) continue

    const currentJoint = newJoints[jointIndex]

    if (currentJoint.type === 'Static') {
      // Static joints: set x, y directly in pylinkage data
      newJoints[jointIndex] = {
        type: 'Static',
        name: jointName,
        x: targetX,
        y: targetY
      } as StaticJoint
    } else if (currentJoint.type === 'Crank') {
      // Crank joints: set meta position, preserve distance and angle
      const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
      newMetaJoints[jointName] = {
        ...currentMeta,
        x: targetX,
        y: targetY
      }
    } else if (currentJoint.type === 'Revolute') {
      // Revolute joints: set meta position, preserve distances
      const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
      newMetaJoints[jointName] = {
        ...currentMeta,
        x: targetX,
        y: targetY
      }
    }
  }

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newJoints
      },
      meta: {
        ...doc.meta,
        joints: newMetaJoints
      }
    },
    movedJoints: jointNames
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MERGE JOINTS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Merge two joints together (source is absorbed into target)
 *
 * @param doc - Current document state
 * @param sourceJoint - Joint to be merged (will be deleted)
 * @param targetJoint - Joint to merge into (will remain)
 * @returns Result with new document and merge details
 */
export function mergeJoints(
  doc: PylinkDocument,
  sourceJoint: string,
  targetJoint: string
): JointMergeResult {
  const result = calculateMergeResult(sourceJoint, targetJoint, doc.meta.links)

  // Update links to point to target instead of source
  const newLinks = { ...doc.meta.links }
  for (const update of result.linksToUpdate) {
    newLinks[update.linkName] = {
      ...newLinks[update.linkName],
      connects: update.newConnects
    }
  }

  // Delete redundant links (self-loops and duplicates)
  for (const linkName of result.linksToDelete) {
    delete newLinks[linkName]
  }

  // Remove the source joint
  const newPylinkageJoints = doc.pylinkage.joints.filter(j => j.name !== sourceJoint)
  const newSolveOrder = doc.pylinkage.solve_order.filter(name => name !== sourceJoint)
  const newJointsMeta = { ...doc.meta.joints }
  delete newJointsMeta[sourceJoint]

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newPylinkageJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newJointsMeta,
        links: newLinks
      }
    },
    sourceJoint,
    targetJoint,
    deletedLinks: result.linksToDelete,
    message: `Merged ${sourceJoint} into ${targetJoint}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENAME JOINT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Rename a joint
 *
 * Updates:
 * - Joint name in pylinkage.joints
 * - References in solve_order
 * - References in meta.joints
 * - References in links
 * - References in other joints (Crank, Revolute)
 *
 * @param doc - Current document state
 * @param oldName - Current joint name
 * @param newName - New joint name
 * @returns Result with new document or error
 */
export function renameJoint(
  doc: PylinkDocument,
  oldName: string,
  newName: string
): RenameResult {
  if (oldName === newName || !newName.trim()) {
    return { newDoc: doc, oldName, newName, success: false, error: 'Invalid name' }
  }

  if (doc.pylinkage.joints.some(j => j.name === newName)) {
    return { newDoc: doc, oldName, newName, success: false, error: `Joint "${newName}" already exists` }
  }

  // Update joint name
  const newJoints = doc.pylinkage.joints.map(j =>
    j.name === oldName ? { ...j, name: newName } : j
  )

  // Update solve_order
  const newSolveOrder = doc.pylinkage.solve_order.map(n => n === oldName ? newName : n)

  // Update meta joints
  const newMetaJoints = { ...doc.meta.joints }
  if (newMetaJoints[oldName]) {
    newMetaJoints[newName] = newMetaJoints[oldName]
    delete newMetaJoints[oldName]
  }

  // Update links that reference this joint
  const newLinks = { ...doc.meta.links }
  Object.entries(doc.meta.links).forEach(([linkName, link]) => {
    newLinks[linkName] = {
      ...link,
      connects: link.connects.map(c => c === oldName ? newName : c)
    }
  })

  // Update joint references in other joints (Crank, Revolute)
  const updatedJoints = newJoints.map(j => {
    if (j.type === 'Crank' && j.joint0?.ref === oldName) {
      return { ...j, joint0: { ref: newName } }
    }
    if (j.type === 'Revolute') {
      const updated = { ...j } as RevoluteJoint
      if (j.joint0?.ref === oldName) updated.joint0 = { ref: newName }
      if (j.joint1?.ref === oldName) updated.joint1 = { ref: newName }
      return updated
    }
    return j
  })

  return {
    newDoc: {
      ...doc,
      pylinkage: { ...doc.pylinkage, joints: updatedJoints, solve_order: newSolveOrder },
      meta: { ...doc.meta, joints: newMetaJoints, links: newLinks }
    },
    oldName,
    newName,
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE JOINT TYPE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Change a joint's type (Static, Crank, Revolute)
 *
 * @param doc - Current document state
 * @param jointName - Name of joint to update
 * @param newType - New joint type
 * @param getJointPosition - Function to get joint positions
 * @param calculateDistance - Function to calculate distance
 * @param getConnectedJoints - Function to find connected joints via links
 * @returns Result with new document
 */
export function updateJointType(
  doc: PylinkDocument,
  jointName: string,
  newType: 'Static' | 'Crank' | 'Revolute',
  getJointPosition: GetJointPositionFn,
  calculateDistance: CalculateDistanceFn,
  getConnectedJoints: (jointName: string, links: Record<string, { connects: string[] }>) => string[]
): UpdatePropertyResult {
  const jointIndex = doc.pylinkage.joints.findIndex(j => j.name === jointName)
  if (jointIndex === -1) {
    return { newDoc: doc, success: false, message: 'Joint not found' }
  }

  // Get current position
  const pos = getJointPosition(jointName)
  if (!pos) {
    return { newDoc: doc, success: false, message: 'Could not determine joint position' }
  }

  const newJoints = [...doc.pylinkage.joints]
  const currentJoint = newJoints[jointIndex]
  const newMetaJoints = { ...doc.meta.joints }
  const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }

  if (newType === 'Static') {
    // Convert to Static: store position in pylinkage data, remove from meta
    newJoints[jointIndex] = {
      type: 'Static',
      name: jointName,
      x: pos[0],
      y: pos[1]
    } as StaticJoint
    newMetaJoints[jointName] = { ...currentMeta, x: undefined, y: undefined }
  } else if (newType === 'Crank') {
    // Find parent joint
    const existingParent = currentJoint.type === 'Crank'
      ? currentJoint.joint0.ref
      : currentJoint.type === 'Revolute'
        ? currentJoint.joint0.ref
        : doc.pylinkage.joints.find(j => j.type === 'Static' && j.name !== jointName)?.name

    if (!existingParent) {
      return { newDoc: doc, success: false, message: 'Need a static joint to reference for Crank type' }
    }

    const parentPos = getJointPosition(existingParent)
    const distance = parentPos ? calculateDistance(parentPos, pos) : 10
    const angle = parentPos ? Math.atan2(pos[1] - parentPos[1], pos[0] - parentPos[0]) : 0

    newJoints[jointIndex] = {
      type: 'Crank',
      name: jointName,
      joint0: { ref: existingParent },
      distance: distance,
      angle: angle
    } as CrankJoint
    newMetaJoints[jointName] = { ...currentMeta, x: pos[0], y: pos[1], show_path: true }
  } else if (newType === 'Revolute') {
    // Find two parent joints from link connections
    const connectedJoints = getConnectedJoints(jointName, doc.meta.links)

    let existingParent0: string | undefined
    let existingParent1: string | undefined

    if (connectedJoints.length >= 2) {
      existingParent0 = connectedJoints[0]
      existingParent1 = connectedJoints[1]
    } else if (currentJoint.type === 'Revolute') {
      existingParent0 = currentJoint.joint0.ref
      existingParent1 = currentJoint.joint1.ref
    } else {
      existingParent0 = currentJoint.type === 'Crank'
        ? currentJoint.joint0.ref
        : doc.pylinkage.joints.find(j => j.name !== jointName)?.name
      existingParent1 = doc.pylinkage.joints.find(
        j => j.name !== jointName && j.name !== existingParent0
      )?.name
    }

    if (!existingParent0 || !existingParent1) {
      return { newDoc: doc, success: false, message: 'Need two joints to reference for Revolute type' }
    }

    const parent0Pos = getJointPosition(existingParent0)
    const parent1Pos = getJointPosition(existingParent1)
    const distance0 = parent0Pos ? calculateDistance(parent0Pos, pos) : 10
    const distance1 = parent1Pos ? calculateDistance(parent1Pos, pos) : 10

    newJoints[jointIndex] = {
      type: 'Revolute',
      name: jointName,
      joint0: { ref: existingParent0 },
      joint1: { ref: existingParent1 },
      distance0: distance0,
      distance1: distance1
    } as RevoluteJoint
    newMetaJoints[jointName] = { ...currentMeta, x: pos[0], y: pos[1], show_path: true }
  }

  return {
    newDoc: {
      ...doc,
      pylinkage: { ...doc.pylinkage, joints: newJoints },
      meta: { ...doc.meta, joints: newMetaJoints }
    },
    success: true,
    message: `Changed ${jointName} to ${newType}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE JOINT META
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Update joint metadata (color, zlevel, show_path, etc.)
 *
 * @param doc - Current document state
 * @param jointName - Name of joint to update
 * @param property - Property to update
 * @param value - New value
 * @returns Result with new document
 */
export function updateJointMeta(
  doc: PylinkDocument,
  jointName: string,
  property: string,
  value: unknown
): UpdatePropertyResult {
  const currentMeta = doc.meta.joints[jointName] || { color: '', zlevel: 0 }

  return {
    newDoc: {
      ...doc,
      meta: {
        ...doc.meta,
        joints: {
          ...doc.meta.joints,
          [jointName]: {
            ...currentMeta,
            [property]: value
          }
        }
      }
    },
    success: true
  }
}
