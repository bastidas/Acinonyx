import React, { useState, useCallback } from 'react'
import { Link, Node, Connection } from '../response_models'

// Graph management utilities
export interface GraphNode extends Node {
  connections: string[] // Array of connection IDs this node participates in
}

export interface GraphConnection extends Connection {
  id: string
}

export interface GraphState {
  nodes: GraphNode[]
  connections: GraphConnection[]
  links: Link[]
}

// Hook for managing graph state
export const useGraphManager = () => {
  const [graphState, setGraphState] = useState<GraphState>({
    nodes: [],
    connections: [],
    links: []
  })

  // Add a new node
  const addNode = useCallback((x: number, y: number, id?: string): GraphNode => {
    // Generate unique ID if not provided, ensuring no collisions
    let nodeId = id
    if (!nodeId) {
      const timestamp = Date.now()
      const random = Math.random().toString(36).substr(2, 9)
      nodeId = `node_${timestamp}_${random}`
      
      // Double-check for uniqueness (very unlikely but safe)
      while (graphState.nodes.some(node => node.id === nodeId)) {
        nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      }
    } else {
      // Ensure provided ID is unique
      while (graphState.nodes.some(node => node.id === nodeId)) {
        const suffix = Math.random().toString(36).substr(2, 4)
        nodeId = `${id}_${suffix}`
      }
    }
    
    const newNode: GraphNode = {
      // Required fields from backend
      name: nodeId, // Use ID as name by default
      n_iterations: 24, // Default to match backend expectation
      init_pos: [x, y], // Set initial position
      // Frontend required fields
      id: nodeId,
      pos: [x, y],
      // Optional fields with defaults
      fixed: false,
      fixed_loc: undefined,
      // GraphNode extension
      connections: []
    }
    
    setGraphState(prev => ({
      ...prev,
      nodes: [...prev.nodes, newNode]
    }))
    
    return newNode
  }, [])

  // Find node at position (within tolerance)
  const findNodeAt = useCallback((x: number, y: number, tolerance: number = 15): GraphNode | null => {
    return graphState.nodes.find(node => {
      const distance = Math.sqrt(
        Math.pow(x - node.pos[0], 2) + Math.pow(y - node.pos[1], 2)
      )
      return distance <= tolerance
    }) || null
  }, [graphState.nodes])

  // Add a connection between two nodes with a link
  const addConnection = useCallback((fromNodeId: string, toNodeId: string, link: Link): string => {
    const connectionId = `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    setGraphState(prev => {
      // Find the connected nodes to check if any are fixed
      const fromNode = prev.nodes.find(n => n.id === fromNodeId)
      const toNode = prev.nodes.find(n => n.id === toNodeId)
      const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
      
      // Update link with correct has_fixed field
      const updatedLink = {
        ...link,
        has_fixed: hasFixedNode
      }
      
      const newConnection: GraphConnection = {
        id: connectionId,
        from_node: fromNodeId,
        to_node: toNodeId,
        link: updatedLink
      }

      // Update nodes to include this connection
      const updatedNodes = prev.nodes.map(node => {
        if (node.id === fromNodeId || node.id === toNodeId) {
          return {
            ...node,
            connections: [...node.connections, connectionId]
          }
        }
        return node
      })

      return {
        ...prev,
        nodes: updatedNodes,
        connections: [...prev.connections, newConnection],
        links: [...prev.links, updatedLink]
      }
    })

    return connectionId
  }, [])

  // Find connections that start or end at a specific point
  const findConnectionsAt = useCallback((x: number, y: number, tolerance: number = 10): GraphConnection[] => {
    return graphState.connections.filter(conn => {
      const fromNode = graphState.nodes.find(n => n.id === conn.from_node)
      const toNode = graphState.nodes.find(n => n.id === conn.to_node)
      
      if (!fromNode || !toNode) return false
      
      const distToStart = Math.sqrt(Math.pow(x - fromNode.pos[0], 2) + Math.pow(y - fromNode.pos[1], 2))
      const distToEnd = Math.sqrt(Math.pow(x - toNode.pos[0], 2) + Math.pow(y - toNode.pos[1], 2))
      
      return distToStart <= tolerance || distToEnd <= tolerance
    })
  }, [graphState.nodes, graphState.connections])

  // Get node by ID
  const getNode = useCallback((nodeId: string): GraphNode | null => {
    return graphState.nodes.find(node => node.id === nodeId) || null
  }, [graphState.nodes])

  // Update a link
  const updateLink = useCallback((linkId: string, updates: Partial<Link>) => {
    setGraphState(prev => ({
      ...prev,
      links: prev.links.map(link => 
        link.id === linkId ? { ...link, ...updates } : link
      )
    }))
  }, [])

  // Update node position and all connected link endpoints
  const updateNodePosition = useCallback((nodeId: string, newX: number, newY: number) => {
    setGraphState(prev => {
      // Update the node position
      const updatedNodes = prev.nodes.map(node => 
        node.id === nodeId ? { ...node, pos: [newX, newY] as [number, number] } : node
      )
      
      // Find all connections involving this node
      const connectionsInvolving = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      
      // Update link endpoints for all connected links
      const updatedLinks = prev.links.map(link => {
        const connection = connectionsInvolving.find(conn => conn.link.id === link.id)
        if (!connection) return link
        
        let updates: Partial<Link> = {}
        
        // Update start_point if this node is the from_node
        if (connection.from_node === nodeId) {
          updates.start_point = [newX, newY] as [number, number]
        }
        
        // Update end_point if this node is the to_node  
        if (connection.to_node === nodeId) {
          updates.end_point = [newX, newY] as [number, number]
        }
        
        return { ...link, ...updates }
      })
      
      return {
        ...prev,
        nodes: updatedNodes,
        links: updatedLinks
      }
    })
  }, [])

  // Delete a link and its connections
  const deleteLink = useCallback((linkId: string) => {
    setGraphState(prev => {
      // Find connections that use this link
      const connectionsToRemove = prev.connections.filter(conn => conn.link.id === linkId)
      const connectionIdsToRemove = connectionsToRemove.map(conn => conn.id)
      
      // Update nodes to remove these connections
      const updatedNodes = prev.nodes.map(node => ({
        ...node,
        connections: node.connections.filter(connId => !connectionIdsToRemove.includes(connId))
      }))
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: prev.connections.filter(conn => conn.link.id !== linkId),
        links: prev.links.filter(link => link.id !== linkId)
      }
    })
  }, [])

  // Delete a node and all its connections
  const deleteNode = useCallback((nodeId: string) => {
    setGraphState(prev => {
      // Find connections that involve this node
      const connectionsToRemove = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      const connectionIdsToRemove = connectionsToRemove.map(conn => conn.id)
      const linkIdsToRemove = connectionsToRemove.map(conn => conn.link.id)
      
      // Update remaining nodes to remove these connections
      const updatedNodes = prev.nodes
        .filter(node => node.id !== nodeId)
        .map(node => ({
          ...node,
          connections: node.connections.filter(connId => !connectionIdsToRemove.includes(connId))
        }))
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: prev.connections.filter(conn => 
          conn.from_node !== nodeId && conn.to_node !== nodeId
        ),
        links: prev.links.filter(link => !linkIdsToRemove.includes(link.id))
      }
    })
  }, [])

  // Merge two nodes - transfer all connections from source to target node
  const mergeNodes = useCallback((sourceNodeId: string, targetNodeId: string) => {
    setGraphState(prev => {
      const sourceNode = prev.nodes.find(node => node.id === sourceNodeId)
      const targetNode = prev.nodes.find(node => node.id === targetNodeId)
      
      if (!sourceNode || !targetNode) return prev
      
      // Update all connections that reference the source node to reference the target node
      const updatedConnections = prev.connections.map(conn => {
        if (conn.from_node === sourceNodeId) {
          return { ...conn, from_node: targetNodeId }
        }
        if (conn.to_node === sourceNodeId) {
          return { ...conn, to_node: targetNodeId }
        }
        return conn
      })
      
      // Update links to use target node position
      const updatedLinks = prev.links.map(link => {
        const connection = updatedConnections.find(conn => conn.link.id === link.id)
        if (!connection) return link
        
        let updates: Partial<Link> = {}
        if (connection.from_node === targetNodeId && link.start_point) {
          updates.start_point = [targetNode.pos[0], targetNode.pos[1]]
        }
        if (connection.to_node === targetNodeId && link.end_point) {
          updates.end_point = [targetNode.pos[0], targetNode.pos[1]]
        }
        
        return { ...link, ...updates }
      })
      
      // Merge connection IDs and remove the source node
      const mergedConnectionIds = [...new Set([...sourceNode.connections, ...targetNode.connections])]
      const updatedNodes = prev.nodes
        .filter(node => node.id !== sourceNodeId)
        .map(node => {
          if (node.id === targetNodeId) {
            return { ...node, connections: mergedConnectionIds }
          }
          return node
        })
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: updatedConnections,
        links: updatedLinks
      }
    })
  }, [])

  // Clear all graph data
  const clearGraph = useCallback(() => {
    setGraphState({
      nodes: [],
      connections: [],
      links: []
    })
  }, [])

  // Get graph structure for backend with all frontend data preserved
  const getGraphStructure = useCallback(() => {
    return {
      nodes: graphState.nodes.map(node => ({
        id: node.id,
        pos: node.pos,
        fixed: node.fixed || false,
        fixed_loc: node.fixed_loc
      })),
      connections: graphState.connections.map(conn => ({
        from_node: conn.from_node,
        to_node: conn.to_node,
        link: conn.link
      })),
      links: graphState.links
    }
  }, [graphState])

  // Get unique z-levels from links
  const getUniqueZLevels = useCallback((): number[] => {
    const zlevels = graphState.links.map(link => link.zlevel || 0)
    return [...new Set(zlevels)].sort((a, b) => a - b)
  }, [graphState.links])

  // Toggle node fixed state and update connected links
  const toggleNodeFixed = useCallback((nodeId: string, fixed: boolean) => {
    setGraphState(prev => {
      // Update the node
      const updatedNodes = prev.nodes.map(node => {
        if (node.id === nodeId) {
          return {
            ...node,
            fixed,
            fixed_loc: fixed ? node.pos : undefined
          }
        }
        return node
      })
      
      // Find all connections involving this node
      const connectionsInvolving = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      
      // Update has_fixed field on connected links
      const updatedLinks = prev.links.map(link => {
        const connection = connectionsInvolving.find(conn => conn.link.id === link.id)
        if (!connection) return link
        
        // Check if either connected node is fixed
        const fromNode = updatedNodes.find(n => n.id === connection.from_node)
        const toNode = updatedNodes.find(n => n.id === connection.to_node)
        const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
        
        return {
          ...link,
          has_fixed: hasFixedNode
        }
      })
      
      return {
        ...prev,
        nodes: updatedNodes,
        links: updatedLinks
      }
    })
  }, [])

  // Utility function to sync has_fixed fields on all links based on connected nodes
  const syncLinkFixedFields = useCallback(() => {
    setGraphState(prev => {
      const updatedLinks = prev.links.map(link => {
        // Find connection for this link
        const connection = prev.connections.find(conn => conn.link.id === link.id)
        if (!connection) return link
        
        // Check if either connected node is fixed
        const fromNode = prev.nodes.find(n => n.id === connection.from_node)
        const toNode = prev.nodes.find(n => n.id === connection.to_node)
        const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
        
        return {
          ...link,
          has_fixed: hasFixedNode
        }
      })
      
      return {
        ...prev,
        links: updatedLinks
      }
    })
  }, [])

  return {
    graphState,
    addNode,
    findNodeAt,
    addConnection,
    findConnectionsAt,
    getNode,
    updateLink,
    updateNodePosition,
    deleteLink,
    deleteNode,
    mergeNodes,
    clearGraph,
    getGraphStructure,
    getUniqueZLevels,
    toggleNodeFixed,
    syncLinkFixedFields
  }
}

// GraphManager component for visual debugging (optional)
export const GraphManager: React.FC<{
  graphState: GraphState
  onNodeClick?: (node: GraphNode) => void
  onConnectionClick?: (connection: GraphConnection) => void
}> = ({ graphState, onNodeClick: _onNodeClick, onConnectionClick: _onConnectionClick }) => {
  return (
    <div style={{ padding: '10px', border: '1px solid #ccc', margin: '10px' }}>
      <h4>Graph State Debug</h4>
      <div>
        <strong>Nodes ({graphState.nodes.length}):</strong>
        <ul>
          {graphState.nodes.map(node => (
            <li key={node.id}>
              {node.id} at ({node.pos[0].toFixed(0)}, {node.pos[1].toFixed(0)}) 
              - {node.connections.length} connections
            </li>
          ))}
        </ul>
      </div>
      <div>
        <strong>Connections ({graphState.connections.length}):</strong>
        <ul>
          {graphState.connections.map(conn => (
            <li key={conn.id}>
              {conn.from_node} → {conn.to_node} via {conn.link.name}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}