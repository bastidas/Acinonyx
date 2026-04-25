import { describe, it, expect } from 'vitest'
import {
  boxFromCorners,
  isFormFullyInsideGroupSelectBox,
  selectionIsExactlyOneMechanism,
  dissociateFormFields,
  DISSOCIATED_FORM_FILL,
  validatePolygonFormAssociations,
  worldPolygonVerticesForForm
} from './formMechanismHelpers'

describe('formMechanismHelpers', () => {
  const box = boxFromCorners(0, 0, 10, 10)
  const linksMeta = {
    L1: { connects: ['A', 'B'] }
  }

  it('isFormFullyInsideGroupSelectBox: all vertices inside for general polygon', () => {
    const inside = isFormFullyInsideGroupSelectBox(
      { points: [[1, 1], [9, 1], [5, 9]], contained_links: ['L1', 'L2'] },
      box,
      () => null,
      linksMeta
    )
    expect(inside).toBe(true)
  })

  it('isFormFullyInsideGroupSelectBox: rejects partial polygon', () => {
    const partial = isFormFullyInsideGroupSelectBox(
      { points: [[1, 1], [11, 1], [5, 5]] },
      box,
      () => null,
      linksMeta
    )
    expect(partial).toBe(false)
  })

  it('isFormFullyInsideGroupSelectBox: single contained link uses endpoints in box', () => {
    const getJ = (j: string) => (j === 'A' ? ([2, 2] as [number, number]) : j === 'B' ? ([8, 8] as [number, number]) : null)
    const ok = isFormFullyInsideGroupSelectBox(
      { points: [[100, 100], [200, 100], [150, 200]], contained_links: ['L1'] },
      box,
      getJ,
      linksMeta
    )
    expect(ok).toBe(true)
  })

  it('isFormFullyInsideGroupSelectBox: mergedLinkName-only uses endpoints', () => {
    const getJ = (j: string) => (j === 'A' ? ([2, 2] as [number, number]) : j === 'B' ? ([8, 8] as [number, number]) : null)
    const ok = isFormFullyInsideGroupSelectBox(
      { points: [[50, 50], [60, 50], [55, 60]], mergedLinkName: 'L1' },
      box,
      getJ,
      linksMeta
    )
    expect(ok).toBe(true)
  })

  it('selectionIsExactlyOneMechanism: true when selection matches one component', () => {
    const links = {
      a: { connects: ['j1', 'j2'], color: '#000' },
      b: { connects: ['j2', 'j3'], color: '#000' }
    }
    const ok = selectionIsExactlyOneMechanism(['j1', 'j2', 'j3'], ['a', 'b'], links)
    expect(ok).toBe(true)
  })

  it('selectionIsExactlyOneMechanism: false for strict subset', () => {
    const links = {
      a: { connects: ['j1', 'j2'], color: '#000' },
      b: { connects: ['j2', 'j3'], color: '#000' }
    }
    const ok = selectionIsExactlyOneMechanism(['j1', 'j2'], ['a'], links)
    expect(ok).toBe(false)
  })

  it('dissociateFormFields clears link fields and sets grey', () => {
    const next = dissociateFormFields({
      contained_links: ['L1'],
      mergedLinkName: 'L1',
      mergedLinkOriginalStart: [0, 0],
      mergedLinkOriginalEnd: [1, 1],
      contained_links_valid: true,
      fillColor: '#f00'
    })
    expect(next.contained_links).toEqual([])
    expect(next.mergedLinkName).toBeUndefined()
    expect(next.mergedLinkOriginalStart).toBeUndefined()
    expect(next.mergedLinkOriginalEnd).toBeUndefined()
    expect(next.contained_links_valid).toBeUndefined()
    expect(next.fillColor).toBe(DISSOCIATED_FORM_FILL)
  })

  it('validatePolygonFormAssociations dissociates when endpoints leave polygon', () => {
    const square: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]]
    const getJ = (j: string) => (j === 'A' ? ([20, 5] as [number, number]) : j === 'B' ? ([20, 6] as [number, number]) : null)
    const meta = { L1: { connects: ['A', 'B'] } }
    const out = validatePolygonFormAssociations(
      [
        {
          type: 'polygon' as const,
          points: square,
          contained_links: ['L1']
        }
      ],
      getJ,
      meta
    )
    expect(out[0]!.contained_links).toEqual([])
  })

  it('validatePolygonFormAssociations keeps merged-link form after rigid translate (stored points unchanged)', () => {
    const origA: [number, number] = [0, 0]
    const origB: [number, number] = [4, 0]
    const stripAroundMergeFrame: [number, number][] = [
      [-0.5, -0.5],
      [4.5, -0.5],
      [4.5, 0.5],
      [-0.5, 0.5]
    ]
    const dx = 10
    const getJ = (j: string) =>
      j === 'A'
        ? ([origA[0] + dx, origA[1]] as [number, number])
        : j === 'B'
          ? ([origB[0] + dx, origB[1]] as [number, number])
          : null
    const meta = { L1: { connects: ['A', 'B'] } }
    const out = validatePolygonFormAssociations(
      [
        {
          type: 'polygon' as const,
          points: stripAroundMergeFrame,
          mergedLinkName: 'L1',
          mergedLinkOriginalStart: origA,
          mergedLinkOriginalEnd: origB,
          contained_links: ['L1'],
          fillColor: '#f00'
        }
      ],
      getJ,
      meta
    )
    expect(out[0]!.mergedLinkName).toBe('L1')
    expect(out[0]!.contained_links).toEqual(['L1'])
    expect(out[0]!.fillColor).toBe('#f00')
  })

  it('validatePolygonFormAssociations dissociates merged form when a second contained link leaves the hull', () => {
    const origA: [number, number] = [0, 0]
    const origB: [number, number] = [4, 0]
    const stripAroundMergeFrame: [number, number][] = [
      [-0.5, -0.5],
      [4.5, -0.5],
      [4.5, 0.5],
      [-0.5, 0.5]
    ]
    const meta = {
      L1: { connects: ['A', 'B'] },
      L2: { connects: ['C', 'D'] }
    }
    const getJ = (j: string) => {
      if (j === 'A') return [0, 0] as [number, number]
      if (j === 'B') return [4, 0] as [number, number]
      if (j === 'C') return [50, 0] as [number, number]
      if (j === 'D') return [54, 0] as [number, number]
      return null
    }
    const out = validatePolygonFormAssociations(
      [
        {
          type: 'polygon' as const,
          points: stripAroundMergeFrame,
          mergedLinkName: 'L1',
          mergedLinkOriginalStart: origA,
          mergedLinkOriginalEnd: origB,
          contained_links: ['L1', 'L2'],
          fillColor: '#f00'
        }
      ],
      getJ,
      meta
    )
    expect(out[0]!.mergedLinkName).toBeUndefined()
    expect(out[0]!.contained_links).toEqual([])
  })

  it('worldPolygonVerticesForForm matches rigid translate of merge-time points', () => {
    const origA: [number, number] = [0, 0]
    const origB: [number, number] = [4, 0]
    const pts: [number, number][] = [
      [-0.5, -0.5],
      [4.5, -0.5],
      [4.5, 0.5],
      [-0.5, 0.5]
    ]
    const meta = { L1: { connects: ['A', 'B'] } }
    const getJ = (j: string) =>
      j === 'A' ? ([10, 0] as [number, number]) : j === 'B' ? ([14, 0] as [number, number]) : null
    const world = worldPolygonVerticesForForm(
      {
        points: pts,
        mergedLinkName: 'L1',
        mergedLinkOriginalStart: origA,
        mergedLinkOriginalEnd: origB
      },
      getJ,
      meta
    )
    expect(world[0]![0]).toBeCloseTo(9.5)
    expect(world[0]![1]).toBeCloseTo(-0.5)
    expect(world[2]![0]).toBeCloseTo(14.5)
    expect(world[2]![1]).toBeCloseTo(0.5)
  })
})
