/**
 * SVG Filter Definitions
 *
 * Defines all the glow filters used for highlighting objects in the canvas.
 * These filters are referenced by url() in stroke/filter properties.
 */

import React from 'react'

/**
 * SVG filter definitions for glow effects
 *
 * Place this component inside an <svg> element as the first child
 */
export const SVGFilters: React.FC = () => {
  return (
    <defs>
      {/* Joint type glows */}
      <filter id="glow-static" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#E74C3C" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-crank" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#F39C12" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-pivot" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#2196F3" floodOpacity="0.8"/>
      </filter>

      {/* Graph color glows for links/objects.
          filterUnits="userSpaceOnUse" avoids clipping on vertical/horizontal lines where
          objectBoundingBox has zero width or height (stroke is ignored for bbox). */}
      <filter id="glow-blue" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#1F77B4" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-orange" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#FF7F0E" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-green" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#2CA02C" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-red" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#D62728" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-purple" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#9467BD" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-brown" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#8C564B" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-pink" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#E377C2" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-gray" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#7F7F7F" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-olive" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#BCBD22" floodOpacity="0.8"/>
      </filter>
      <filter id="glow-cyan" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#17BECF" floodOpacity="0.8"/>
      </filter>

      {/* Move group glow (grey) - special case */}
      <filter id="glow-movegroup" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#9E9E9E" floodOpacity="0.6"/>
      </filter>

      {/* Merge highlight glow (cyan) - special case */}
      <filter id="glow-merge" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#00BCD4" floodOpacity="0.9"/>
      </filter>
    </defs>
  )
}

export default SVGFilters
