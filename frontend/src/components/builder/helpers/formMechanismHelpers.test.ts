import { describe, it, expect } from 'vitest'
import {
  boxFromCorners,
  isFormFullyInsideGroupSelectBox,
  selectionIsExactlyOneMechanism,
  dissociateFormFields,
  DISSOCIATED_FORM_FILL,
  validatePolygonFormAssociations
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
})
