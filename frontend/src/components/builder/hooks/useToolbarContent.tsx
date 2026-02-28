/**
 * useToolbarContent
 *
 * Returns a function (id: string) => ReactNode that renders the appropriate toolbar content.
 * Single place for the toolbar-by-id switch; BuilderTab only wires props and passes the getter.
 */

import React, { useCallback } from 'react'
import {
  ToolsToolbar,
  LinksToolbar,
  NodesToolbar,
  MoreToolbar,
  FormsToolbar,
  SettingsToolbar,
  OptimizationToolbar
} from '../toolbars'
import type { ToolsToolbarProps } from '../toolbars/ToolsToolbar'
import type { LinksToolbarProps } from '../toolbars/LinksToolbar'
import type { NodesToolbarProps } from '../toolbars/NodesToolbar'
import type { MoreToolbarProps } from '../toolbars/MoreToolbar'
import type { FormsToolbarProps } from '../toolbars/FormsToolbar'
import type { SettingsToolbarProps } from '../toolbars/SettingsToolbar'
import type { OptimizationToolbarProps } from '../toolbars/OptimizationToolbar'

export interface UseToolbarContentParams {
  toolsProps: ToolsToolbarProps
  linksProps: LinksToolbarProps
  nodesProps: NodesToolbarProps
  moreProps: MoreToolbarProps
  formsProps: FormsToolbarProps
  settingsProps: SettingsToolbarProps
  optimizeProps: OptimizationToolbarProps
}

export type GetToolbarContent = (id: string) => React.ReactNode

/**
 * Returns a stable function that renders toolbar content by id.
 * Pass a memoized params object (e.g. useMemo) so the returned function identity is stable.
 */
export function useToolbarContent(params: UseToolbarContentParams): GetToolbarContent {
  const {
    toolsProps,
    linksProps,
    nodesProps,
    moreProps,
    formsProps,
    settingsProps,
    optimizeProps
  } = params

  return useCallback(
    (id: string): React.ReactNode => {
      switch (id) {
        case 'tools':
          return <ToolsToolbar {...toolsProps} />
        case 'links':
          return <LinksToolbar {...linksProps} />
        case 'nodes':
          return <NodesToolbar {...nodesProps} />
        case 'more':
          return <MoreToolbar {...moreProps} />
        case 'forms':
          return <FormsToolbar {...formsProps} />
        case 'settings':
          return <SettingsToolbar {...settingsProps} />
        case 'optimize':
          return <OptimizationToolbar {...optimizeProps} />
        default:
          return null
      }
    },
    [
      toolsProps,
      linksProps,
      nodesProps,
      moreProps,
      formsProps,
      settingsProps,
      optimizeProps
    ]
  )
}
