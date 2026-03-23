import { describe, it, expect } from 'vitest'
import {
  remapEdgeReferencesInDrawnObjects,
  removeDrawnObjectsReferencingDeletedEdges
} from './drawnObjectsSync'

describe('remapEdgeReferencesInDrawnObjects', () => {
  it('remaps contained_links and mergedLinkName for a merged polygon', () => {
    const objects = [
      {
        id: 'p1',
        contained_links: ['a', 'b', 'c'],
        mergedLinkName: 'a'
      }
    ]
    const out = remapEdgeReferencesInDrawnObjects(objects, 'a', 'alpha') as typeof objects
    expect(out[0].contained_links).toEqual(['alpha', 'b', 'c'])
    expect(out[0].mergedLinkName).toBe('alpha')
  })

  it('remaps only contained_links when mergedLinkName differs', () => {
    const objects = [{ id: 'p1', contained_links: ['x', 'y'], mergedLinkName: 'y' }]
    const out = remapEdgeReferencesInDrawnObjects(objects, 'x', 'x2') as typeof objects
    expect(out[0].contained_links).toEqual(['x2', 'y'])
    expect(out[0].mergedLinkName).toBe('y')
  })

  it('returns a new array and leaves unrelated objects unchanged by reference', () => {
    const other = { id: 'o', foo: 1 }
    const objects = [other, { id: 'p', contained_links: ['e'] }]
    const out = remapEdgeReferencesInDrawnObjects(objects, 'e', 'e2')
    expect(out).not.toBe(objects)
    expect(out[0]).toBe(other)
  })

  it('no-ops when old and new ids are equal', () => {
    const objects = [{ id: 'p', contained_links: ['same'] }]
    const out = remapEdgeReferencesInDrawnObjects(objects, 'same', 'same')
    expect(out).toEqual([{ id: 'p', contained_links: ['same'] }])
  })
})

describe('removeDrawnObjectsReferencingDeletedEdges', () => {
  it('removes polygon when mergedLinkName is deleted', () => {
    const objects = [{ id: 'p1', mergedLinkName: 'gone', contained_links: ['gone'] }]
    const { objects: out, removedIds } = removeDrawnObjectsReferencingDeletedEdges(objects, new Set(['gone']))
    expect(out).toHaveLength(0)
    expect(removedIds).toEqual(['p1'])
  })

  it('removes polygon when any contained_links member is deleted', () => {
    const objects = [{ id: 'p1', mergedLinkName: 'a', contained_links: ['a', 'b', 'c'] }]
    const { objects: out, removedIds } = removeDrawnObjectsReferencingDeletedEdges(objects, new Set(['b']))
    expect(out).toHaveLength(0)
    expect(removedIds).toEqual(['p1'])
  })

  it('keeps objects when deleted set is empty', () => {
    const objects = [{ id: 'p1', contained_links: ['a'] }]
    const { objects: out, removedIds } = removeDrawnObjectsReferencingDeletedEdges(objects, new Set())
    expect(out).toHaveLength(1)
    expect(removedIds).toEqual([])
  })

  it('keeps unrelated polygons', () => {
    const objects = [
      { id: 'drop', contained_links: ['x'] },
      { id: 'keep', contained_links: ['y'] }
    ]
    const { objects: out, removedIds } = removeDrawnObjectsReferencingDeletedEdges(objects, new Set(['x']))
    expect(out.map(o => (o as { id: string }).id)).toEqual(['keep'])
    expect(removedIds).toEqual(['drop'])
  })
})
