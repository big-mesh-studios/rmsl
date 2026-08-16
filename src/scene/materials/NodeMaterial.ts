import {
  Fn, mat3, output, vec4,
  type Node,
} from "../../rmsl";
import { Material } from "./Material";
import {
  Builder,
  type AnySamplerBinding,
  type AttributeBinding, type UniformBinding, type VaryingBinding,
} from "./nodes/Builder";
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
export function resolveSlot<T extends Node<any>>(
  slot: SlotValue<T> | undefined,
  b: Builder,
): T | undefined {
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
    const position = resolveSlot(this.positionNode, b) ?? b.position;
    const position4 = vec4(position, 1);
    // The instance matrix applies first, in local space, and the mesh's own
    // model matrix follows — for a plain mesh the instance transform is absent.
    const localPosition = b.instancing ? b.instanceMatrix.mul(position4) : position4;
    const worldPosition = b.modelMatrix.mul(localPosition);
    b.positionWorld.assign(worldPosition.xyz);
    let normal = resolveSlot(this.normalNode, b) ?? b.normal;
    if (b.instancing) normal = mat3(b.instanceMatrix).mul(normal);
    b.normalWorld.assign(b.normalMatrix.mul(normal).normalize());
    b.uvVarying.assign(b.uv);
    if (b.instancingColor) b.instanceColorVarying.assign(b.instanceColor);
    return b.projectionMatrix.mul(b.viewMatrix.mul(worldPosition));
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
  build(
    scene: Scene,
    options: { instancing?: boolean; instancingColor?: boolean } = {},
  ): MaterialProgram {
    const b = new Builder();
    b.instancing = options.instancing ?? false;
    b.instancingColor = options.instancingColor ?? false;
    this.setup(b, scene);

    b.stage = "vertex";
    const vertex = Fn(() => this.vertexNode ? this.vertexNode(b) : this.buildVertexBody(b))() as Node<"vec4">;
    b.stage = "fragment";
    const fragment = Fn(() => {
      const outColor = output("vec4");
      const color = this.fragmentNode ? this.fragmentNode(b) : this.buildFragmentBody(b);
      // The per-instance color tints the material's color, whatever it is.
      const tinted = b.instancingColor
        ? vec4(color.rgb.mul(b.instanceColorVarying), color.a)
        : color;
      outColor.assign(tinted);
      return outColor;
    })() as Node<"vec4">;

    const v = collectNodes(vertex);
    const f = collectNodes(fragment);
    const usedUniforms = new Set([...v.uniforms, ...f.uniforms]);
    const usedAttributes = new Set([...v.attributes, ...f.attributes]);
    const usedVaryings = new Set([...v.varyings, ...f.varyings]);

    return {
      vertexRoot: vertex,
      fragmentRoot: fragment,
      uniforms: [...b.uniforms.values()]
        .filter((binding) => usedUniforms.has(binding.node)),
      attributes: [...b.attributes.values()]
        .filter((binding) => usedAttributes.has(binding.node)),
      varyings: [...b.varyings.values()]
        .filter((binding) => usedVaryings.has(binding.node)),
      samplers: [...b.samplers.values()],
    };
  }
}
