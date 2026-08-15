import { float, vec2, vec3, type Node } from "../rmsl";

/**
 * A number or a float node. Narrower than the core `FloatLike`, which also
 * admits a bare `BaseNode` without the operation methods — this is what the
 * effects' parameters and the core free functions both accept.
 */
export type FloatIn = number | Node<"float">;

/** A number or an int node, for the effects' integer parameters. */
export type IntIn = number | Node<"int">;

/** A 2D position: an array or a vec2 node. */
export type Vec2In = [number, number] | Node<"vec2">;

/** A 3D position: an array or a vec3 node. */
export type Vec3In = [number, number, number] | Node<"vec3">;

/** A float 2D sampler node carrying the sampling operations. */
export type Sampler2D = Node<"sampler2D">;

/** A float 3D sampler node carrying the sampling operations. */
export type Sampler3D = Node<"sampler3D">;

/**
 * Lift a number or a float node to a float node. The public constructors only
 * admit numbers and int nodes, so a float node has to pass through by hand.
 */
export function f(x: FloatIn): Node<"float"> {
  return typeof x === "number" ? float(x) : x;
}

/** Resolve a `Vec2In` to a vec2 node. */
export function vec2In(v: Vec2In): Node<"vec2"> {
  return Array.isArray(v) ? vec2(v[0], v[1]) : v;
}

/** Resolve a `Vec3In` to a vec3 node. */
export function vec3In(v: Vec3In): Node<"vec3"> {
  return Array.isArray(v) ? vec3(v[0], v[1], v[2]) : v;
}
