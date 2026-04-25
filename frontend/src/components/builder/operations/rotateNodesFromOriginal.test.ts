/**
 * Rotation invariance for rigid group moves (hypergraph / LinkageDocument).
 */

import { describe, it, expect } from 'vitest'
import type { LinkageDocument, Edge, Node } from '../../../types'
import { rotateNodesFromOriginal } from './nodeOperations'

function dist(a: [number, number], b: [number, number]): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

function edgeEndpointDistance(doc: LinkageDocument, edge: Edge): number {
  const s = doc.linkage.nodes[edge.source]?.position
  const t = doc.linkage.nodes[edge.target]?.position
  if (!s || !t) return NaN
  return dist(s as [number, number], t as [number, number])
}

function makeOpenChainDoc(): LinkageDocument {
  return {
    name: 'test-doc',
    version: '1',
    linkage: {
      name: 'chain',
      nodes: {
        A: {
          id: 'A',
          position: [0, 0],
          role: 'fixed',
          jointType: 'revolute',
          name: 'A'
        },
        B: {
          id: 'B',
          position: [1, 0],
          role: 'follower',
          jointType: 'revolute',
          angle: 0.25,
          initialAngle: 0.1,
          name: 'B'
        },
        C: {
          id: 'C',
          position: [2, 0.5],
          role: 'follower',
          jointType: 'revolute',
          name: 'C'
        }
      },
      edges: {
        link_AB: { id: 'link_AB', source: 'A', target: 'B', distance: 1 },
        link_BC: { id: 'link_BC', source: 'B', target: 'C', distance: Math.hypot(1, 0.5) }
      },
      hyperedges: {}
    },
    meta: {
      nodes: {
        A: { color: '#111111', zlevel: 0, metaValue: 'pin' },
        B: { color: '#222222', zlevel: 1, showPath: true },
        C: { color: '#333333', zlevel: 0 }
      },
      edges: {
        link_AB: { color: '#aaaaaa', isGround: true, zlevel: 0 },
        link_BC: { color: '#bbbbbb', zlevel: 2 }
      }
    }
  }
}

function snapshotNodeKinematics(nodes: Record<string, Node>) {
  const out: Record<
    string,
    Pick<Node, 'role' | 'jointType' | 'angle' | 'initialAngle'>
  > = {}
  for (const [id, n] of Object.entries(nodes)) {
    out[id] = {
      role: n.role,
      jointType: n.jointType,
      angle: n.angle,
      initialAngle: n.initialAngle
    }
  }
  return out
}

function snapshotEdgeDistances(edges: Record<string, Edge>) {
  const out: Record<string, number> = {}
  for (const [id, e] of Object.entries(edges)) {
    out[id] = e.distance
  }
  return out
}

describe('rotateNodesFromOriginal', () => {
  it('rotates 90° about centroid, preserves edge distance fields, roles, angles, and meta', () => {
    const before = makeOpenChainDoc()
    const originalPositions: Record<string, [number, number]> = {}
    for (const id of Object.keys(before.linkage.nodes)) {
      const p = before.linkage.nodes[id]!.position
      originalPositions[id] = [p[0], p[1]]
    }
    const pivot: [number, number] = [1, 1 / 6] // centroid of triangle (0,0),(1,0),(2,0.5)
    const angle = Math.PI / 2

    const { doc: after } = rotateNodesFromOriginal(before, originalPositions, pivot, angle)

    expect(snapshotEdgeDistances(after.linkage.edges)).toEqual(
      snapshotEdgeDistances(before.linkage.edges)
    )
    expect(snapshotNodeKinematics(after.linkage.nodes)).toEqual(
      snapshotNodeKinematics(before.linkage.nodes)
    )
    expect(after.meta).toEqual(before.meta)

    for (const edge of Object.values(after.linkage.edges)) {
      const geom = edgeEndpointDistance(after, edge)
      expect(geom).toBeCloseTo(edge.distance, 10)
    }
  })

  it('preserves pairwise distances for every node pair (rigid rotation)', () => {
    const before = makeOpenChainDoc()
    const ids = Object.keys(before.linkage.nodes)
    const originalPositions: Record<string, [number, number]> = {}
    for (const id of ids) {
      const p = before.linkage.nodes[id]!.position
      originalPositions[id] = [p[0], p[1]]
    }
    const pivot: [number, number] = [5, -3]
    const angle = 0.7

    const pairwiseBefore: number[] = []
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        pairwiseBefore.push(
          dist(originalPositions[ids[i]!]!, originalPositions[ids[j]!]!)
        )
      }
    }

    const { doc: after } = rotateNodesFromOriginal(before, originalPositions, pivot, angle)

    const posAfter = (id: string) => after.linkage.nodes[id]!.position as [number, number]
    let k = 0
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const d = dist(posAfter(ids[i]!), posAfter(ids[j]!))
        expect(d).toBeCloseTo(pairwiseBefore[k]!, 10)
        k++
      }
    }
  })

  it('returns same doc when angle is 0', () => {
    const before = makeOpenChainDoc()
    const originalPositions: Record<string, [number, number]> = {
      A: [0, 0],
      B: [1, 0]
    }
    const { doc: after } = rotateNodesFromOriginal(before, originalPositions, [0, 0], 0)
    expect(after).toBe(before)
  })
})
