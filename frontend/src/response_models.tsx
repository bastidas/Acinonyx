// Type definitions for Acinonyx frontend

// Frontend-specific metadata for links (not sent to backend)
export interface LinkMeta {
  id: string // Generated on frontend, required for UI
  start_point?: [number, number] // For rendering link position
  end_point?: [number, number] // For rendering link position
  color?: string // Display color
}

export interface Link {
  // Required fields matching backend ground truth
  name: string // Required in backend
  length: number // Required, must be > 0 and <= 100
  n_iterations: number // Required, must be >= 1 and <= 1000
  has_fixed: boolean // Required in backend
  
  // Optional fields from backend
  target_length?: number | null
  target_cost_func?: string | null
  fixed_loc?: [number, number] | null
  has_constraint?: boolean
  is_driven?: boolean
  is_ground?: boolean // Ground link between fixed anchors (frame)
  flip?: boolean
  zlevel?: number
  
  // Frontend metadata (single source of truth for UI properties)
  meta: LinkMeta
}

export interface Node {
  // Required fields matching backend ground truth
  name: string // Required in backend
  n_iterations: number // Required, must be >= 1 and <= 1000
  
  // Optional fields from backend
  fixed?: boolean
  fixed_loc?: [number, number] | null
  target_loc?: [number, number] | null
  target_radius?: number | null
  target_cost_func?: string | null
  init_pos: [number, number] | null

  // Frontend-specific properties for UI (required for frontend)
  id: string // Generated on frontend, required for UI
  pos: [number, number] // For UI positioning, required for UI
}


export interface GraphStructure {
  nodes: Node[]
  connections: Connection[]
  links: Link[]
}

export interface CreateLinkRequest {
  // Required fields
  name: string
  length: number
  n_iterations: number
  has_fixed: boolean
  
  // Optional fields
  target_length?: number | null
  target_cost_func?: string | null
  fixed_loc?: [number, number] | null
  has_constraint?: boolean
  is_driven?: boolean
  flip?: boolean
  zlevel?: number
}

export interface ModifyLinkRequest {
  id: string
  property: string
  value: any
}

// Connection is now a lightweight reference - link data lives in links array only
export interface Connection {
  from_node: string
  to_node: string
  link_id: string // Reference to link by meta.id - NOT an embedded link object
}



export interface StatusResponse {
  status: string
  message?: string
}

// Helper type for extracting backend-only fields from Link (excludes meta)
export type BackendLink = Omit<Link, 'meta'>
