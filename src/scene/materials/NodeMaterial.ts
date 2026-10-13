import { Fn, vec4, type Node } from "../../rmsl";
import { Material } from "./Material";
import {
  Builder,
  type AnySamplerBinding,
  type AttributeBinding,
  type UniformBinding,
  type VaryingBinding,
} from "./nodes/Builder";
import { linkStages } from "../../backends/shared";
import { collectNodes } from "./nodes/graph";
import type { Scene } from "../scenes/Scene";

export type SlotValue<T extends Node<any>> = T | ((b: Builder) => T);

/**
 * The result of compiling a material: the vertex and fragment roots plus the
 * exact set of uniforms/attributes/varyings the compiled shaders reference.
 */
export interface MaterialProgram {
  vertexRoot: Node<"vec4">;
  fragmentRoot: Node<"vec4">;
  uniforms: UniformBinding[];
  attributes: AttributeBinding[];
  varyings: VaryingBinding[];
  samplers: AnySamplerBinding[];
}

/** Evaluate a material slot, running builder functions in the active pass. */
export function resolveSlot<T extends Node<any>>(slot: SlotValue<T> | undefined, b: Builder): T | undefined {
  if (slot === undefined) return undefined;
  return typeof slot === "function" ? (slot as (b: Builder) => T)(b) : slot;
}

/**
 * The base of the node-based materials. Subclasses register their property
 * uniforms in `setup` and return a fragment node graph from
 * `buildFragmentBody`. The vertex stage is shared: it passes position, normal
 * and uv to world space and returns the clip-space position, honoring a
 * `positionNode` slot.
 */
export class NodeMaterial extends Material {
  readonly isNodeMaterial = true;

  colorNode?: SlotValue<Node<"vec3">>;
  opacityNode?: SlotValue<Node<"float">>;
  roughnessNode?: SlotValue<Node<"float">>;
  metalnessNode?: SlotValue<Node<"float">>;
  emissiveNode?: SlotValue<Node<"vec3">>;
  normalNode?: SlotValue<Node<"vec3">>;
  positionNode?: SlotValue<Node<"vec3">>;
  uvNode?: SlotValue<Node<"vec2">>;

  /** Full control over the vertex stage; must return clip space. */
  vertexNode?: (b: Builder) => Node<"vec4">;
  /** Full control over the fragment stage; returns the final color. */
  fragmentNode?: (b: Builder) => Node<"vec4">;

  protected setup(_b: Builder, _scene: Scene): void {}

  protected buildVertexBody(b: Builder): Node<"vec4"> {
    // The builder's position and normal carry the instance matrix already, and
    // the mesh's own model matrix follows.
    const worldPosition = b.modelMatrix.mul(vec4(this.localPosition(b), 1));
    b.positionWorld.assign(worldPosition.xyz);
    b.normalWorld.assign(b.normalMatrix.mul(b.normal).normalize());
    b.uvVarying.assign(b.uv);
    if (b.instancingColor) b.instanceColorVarying.assign(b.instanceColor);
    return b.projectionMatrix.mul(b.viewMatrix.mul(worldPosition));
  }

  private readonly localPositions = new WeakMap<Builder, Node<"vec3">>();

  /**
   * The object-space position the vertex stage ends with, as TSL's
   * `positionLocal`: `positionNode`, or else the builder's position.
   */
  protected localPosition(b: Builder): Node<"vec3"> {
    let position = this.localPositions.get(b);
    if (!position) {
      position = resolveSlot(this.positionNode, b) ?? b.position;
      this.localPositions.set(b, position);
    }
    return position;
  }

  /** Returns the final color (a vec4, alpha included) for this material. */
  protected buildFragmentBody(_b: Builder): Node<"vec4"> {
    return vec4(1, 1, 1, 1);
  }

  /**
   * Compiles this material's graph into a `MaterialProgram`. `options`
   * describe the drawable the program is being built for: an `InstancedMesh`
   * (`instancing`) and whether it carries per-instance colors
   * (`instancingColor`) pull the corresponding instanced attributes and the
   * instance transform into the shaders. The renderer derives them per object;
   * a plain mesh compiles the same graph without them.
   */
  build(scene: Scene, options: { instancing?: boolean; instancingColor?: boolean } = {}): MaterialProgram {
    const b = new Builder();
    b.instancing = options.instancing ?? false;
    b.instancingColor = options.instancingColor ?? false;
    this.setup(b, scene);

    // The fragment stage is built first, so the vertex stage knows which
    // object-space attributes to pass it.
    b.stage = "fragment";
    const fragment = Fn(() => {
      const color = this.fragmentNode ? this.fragmentNode(b) : this.buildFragmentBody(b);
      // The per-instance color tints the material's color, whatever it is.
      return b.instancingColor ? vec4(color.rgb.mul(b.instanceColorVarying), color.a) : color;
    })() as Node<"vec4">;
    b.stage = "vertex";
    const vertex = Fn(() => {
      const clip = this.vertexNode ? this.vertexNode(b) : this.buildVertexBody(b);
      // The fragment stage reads the position the vertex stage ends with; the compiler passes on any attribute it reads.
      if (b.fragmentReads.has("position")) b.varying("positionLocal", "vec3").assign(this.localPosition(b));
      return clip;
    })() as Node<"vec4">;

    const linked = linkStages(vertex, fragment);
    const v = collectNodes(linked.vertex);
    const f = collectNodes(linked.fragment);
    const usedUniforms = new Set([...v.uniforms, ...f.uniforms]);
    const usedAttributes = new Set([...v.attributes, ...f.attributes]);
    const usedVaryings = new Set([...v.varyings, ...f.varyings]);
    // An attribute the fragment stage reads reaches it in a varying the compiler made, which the program names after the attribute.
    const varyings = [...b.varyings.values()].filter((binding) => usedVaryings.has(binding.node));
    for (const attribute of b.attributes.values()) {
      const carrier = linked.carriers.get(attribute.node.name);
      if (carrier && !varyings.some((binding) => binding.name === attribute.name)) {
        varyings.push({ node: carrier, name: attribute.name });
      }
    }

    return {
      vertexRoot: linked.vertex,
      fragmentRoot: linked.fragment,
      uniforms: [...b.uniforms.values()].filter((binding) => usedUniforms.has(binding.node)),
      attributes: [...b.attributes.values()].filter((binding) => usedAttributes.has(binding.node)),
      varyings,
      samplers: [...b.samplers.values()],
    };
  }
}
