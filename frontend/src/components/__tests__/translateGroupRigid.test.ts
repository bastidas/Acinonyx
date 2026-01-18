/**
 * Tests for rigid body translation of joint groups
 * 
 * These tests verify that when moving a group of joints:
 * 1. All joints move by the same delta (dx, dy)
 * 2. Joint types (Static, Crank, Revolute) are preserved
 * 3. Structural properties (distances, angles, references) are NOT changed
 * 4. Only positional data is modified
 */

// Types matching the main application
interface JointRef {
  ref: string
}

interface StaticJoint {
  type: 'Static'
  name: string
  x: number
  y: number
}

interface CrankJoint {
  type: 'Crank'
  name: string
  joint0: JointRef
  distance: number
  angle: number
}

interface RevoluteJoint {
  type: 'Revolute'
  name: string
  joint0: JointRef
  joint1: JointRef
  distance0: number
  distance1: number
}

type PylinkJoint = StaticJoint | CrankJoint | RevoluteJoint

interface JointMeta {
  color?: string
  zlevel?: number
  x?: number
  y?: number
}

// Pure function version of translateGroupRigid for testing
function translateGroupRigidPure(
  joints: PylinkJoint[],
  metaJoints: Record<string, JointMeta>,
  jointNames: string[],
  originalPositions: Record<string, [number, number]>,
  dx: number,
  dy: number
): { joints: PylinkJoint[], metaJoints: Record<string, JointMeta> } {
  const newJoints = [...joints]
  const newMetaJoints = { ...metaJoints }
  
  for (const jointName of jointNames) {
    const originalPos = originalPositions[jointName]
    if (!originalPos) continue
    
    const targetX = originalPos[0] + dx
    const targetY = originalPos[1] + dy
    
    const jointIndex = newJoints.findIndex(j => j.name === jointName)
    if (jointIndex === -1) continue
    
    const currentJoint = newJoints[jointIndex]
    
    if (currentJoint.type === 'Static') {
      newJoints[jointIndex] = {
        type: 'Static',
        name: jointName,
        x: targetX,
        y: targetY
      }
    } else if (currentJoint.type === 'Crank') {
      const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
      newMetaJoints[jointName] = {
        ...currentMeta,
        x: targetX,
        y: targetY
      }
    } else if (currentJoint.type === 'Revolute') {
      const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
      newMetaJoints[jointName] = {
        ...currentMeta,
        x: targetX,
        y: targetY
      }
    }
  }
  
  return { joints: newJoints, metaJoints: newMetaJoints }
}

// Helper to calculate distance between two points
function distance(p1: [number, number], p2: [number, number]): number {
  return Math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
}

describe('translateGroupRigid', () => {
  // Test fixture: A simple 4-bar linkage
  const createFourBarLinkage = () => {
    const joints: PylinkJoint[] = [
      { type: 'Static', name: 'ground1', x: 0, y: 0 },
      { type: 'Static', name: 'ground2', x: 4, y: 0 },
      { type: 'Crank', name: 'crank_end', joint0: { ref: 'ground1' }, distance: 1, angle: 0.1 },
      { type: 'Revolute', name: 'coupler_end', joint0: { ref: 'crank_end' }, joint1: { ref: 'rocker_end' }, distance0: 3, distance1: 2 },
      { type: 'Revolute', name: 'rocker_end', joint0: { ref: 'ground2' }, joint1: { ref: 'coupler_end' }, distance0: 2, distance1: 2 }
    ]
    
    const metaJoints: Record<string, JointMeta> = {
      'ground1': { color: '#ff0000', zlevel: 0 },
      'ground2': { color: '#ff0000', zlevel: 0 },
      'crank_end': { color: '#00ff00', zlevel: 0, x: 1, y: 0 },
      'coupler_end': { color: '#0000ff', zlevel: 0, x: 3, y: 1 },
      'rocker_end': { color: '#0000ff', zlevel: 0, x: 4, y: 2 }
    }
    
    // Original positions before drag
    const originalPositions: Record<string, [number, number]> = {
      'ground1': [0, 0],
      'ground2': [4, 0],
      'crank_end': [1, 0],
      'coupler_end': [3, 1],
      'rocker_end': [4, 2]
    }
    
    return { joints, metaJoints, originalPositions }
  }

  test('should translate all Static joints by the same delta', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = ['ground1', 'ground2']
    const dx = 5
    const dy = 3
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, dx, dy)
    
    const ground1 = result.joints.find(j => j.name === 'ground1') as StaticJoint
    const ground2 = result.joints.find(j => j.name === 'ground2') as StaticJoint
    
    expect(ground1.x).toBe(0 + dx)
    expect(ground1.y).toBe(0 + dy)
    expect(ground2.x).toBe(4 + dx)
    expect(ground2.y).toBe(0 + dy)
  })

  test('should preserve joint types after translation', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, 10, 10)
    
    expect(result.joints.find(j => j.name === 'ground1')?.type).toBe('Static')
    expect(result.joints.find(j => j.name === 'crank_end')?.type).toBe('Crank')
    expect(result.joints.find(j => j.name === 'coupler_end')?.type).toBe('Revolute')
  })

  test('should preserve Crank joint structural properties (distance, angle)', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    
    const crankBefore = joints.find(j => j.name === 'crank_end') as CrankJoint
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, 10, 10)
    
    const crankAfter = result.joints.find(j => j.name === 'crank_end') as CrankJoint
    
    expect(crankAfter.distance).toBe(crankBefore.distance)
    expect(crankAfter.angle).toBe(crankBefore.angle)
    expect(crankAfter.joint0.ref).toBe(crankBefore.joint0.ref)
  })

  test('should preserve Revolute joint structural properties (distance0, distance1)', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    
    const revBefore = joints.find(j => j.name === 'coupler_end') as RevoluteJoint
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, 10, 10)
    
    const revAfter = result.joints.find(j => j.name === 'coupler_end') as RevoluteJoint
    
    expect(revAfter.distance0).toBe(revBefore.distance0)
    expect(revAfter.distance1).toBe(revBefore.distance1)
    expect(revAfter.joint0.ref).toBe(revBefore.joint0.ref)
    expect(revAfter.joint1.ref).toBe(revBefore.joint1.ref)
  })

  test('should update meta positions for Crank and Revolute joints', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    const dx = 5
    const dy = 3
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, dx, dy)
    
    // Check Crank meta position
    expect(result.metaJoints['crank_end'].x).toBe(1 + dx)
    expect(result.metaJoints['crank_end'].y).toBe(0 + dy)
    
    // Check Revolute meta position
    expect(result.metaJoints['coupler_end'].x).toBe(3 + dx)
    expect(result.metaJoints['coupler_end'].y).toBe(1 + dy)
  })

  test('should preserve relative distances between joints after translation', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    
    // Calculate original distance between ground1 and ground2
    const originalDist = distance(originalPositions['ground1'], originalPositions['ground2'])
    
    const dx = 100
    const dy = -50
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, dx, dy)
    
    const ground1After = result.joints.find(j => j.name === 'ground1') as StaticJoint
    const ground2After = result.joints.find(j => j.name === 'ground2') as StaticJoint
    
    const newDist = distance([ground1After.x, ground1After.y], [ground2After.x, ground2After.y])
    
    // Distance should be preserved (within floating point tolerance)
    expect(newDist).toBeCloseTo(originalDist, 10)
  })

  test('should handle zero delta (no movement)', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    const jointNames = Object.keys(originalPositions)
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, 0, 0)
    
    const ground1 = result.joints.find(j => j.name === 'ground1') as StaticJoint
    expect(ground1.x).toBe(0)
    expect(ground1.y).toBe(0)
  })

  test('should only move joints in the specified jointNames array', () => {
    const { joints, metaJoints, originalPositions } = createFourBarLinkage()
    // Only move ground1, leave ground2 unchanged
    const jointNames = ['ground1']
    
    const result = translateGroupRigidPure(joints, metaJoints, jointNames, originalPositions, 10, 10)
    
    const ground1 = result.joints.find(j => j.name === 'ground1') as StaticJoint
    const ground2 = result.joints.find(j => j.name === 'ground2') as StaticJoint
    
    expect(ground1.x).toBe(10)  // Moved
    expect(ground1.y).toBe(10)
    expect(ground2.x).toBe(4)   // Not moved (original position)
    expect(ground2.y).toBe(0)
  })
})

