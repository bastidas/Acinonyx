// Type definitions for Acinonyx frontend

export interface Link {
  id: string
  length: number // Always ensure this is a number
  name?: string
  n_iterations: number
  fixed_loc?: [number, number] | null
  has_fixed: boolean
  has_constraint: boolean
  is_driven: boolean
  flip: boolean
  zlevel: number
  // Frontend-specific properties for UI
  start_point?: [number, number]
  end_point?: [number, number]
  color?: string
}

export interface Node {
  id: string
  // x: number
  // y: number
  pos: [number, number]
  fixed?: boolean
  fixed_loc?: [number, number]
}



export interface CreateLinkRequest {
  length: number
  name?: string
  n_iterations?: number
  fixed_loc?: [number, number]
  is_driven?: boolean
  flip?: boolean
}

export interface ModifyLinkRequest {
  id: string
  property: string
  value: any
}

export interface Connection {
  from_node: string
  to_node: string
  link: Link
}

export interface GraphStructure {
  nodes: Node[]
  connections: Connection[]
  links: Link[]
}

export interface StatusResponse {
  status: string
  message?: string
}