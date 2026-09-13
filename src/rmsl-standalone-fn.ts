// === Standalone function compilers (for Three.js glslFn/wgslFn embedding) ===
import { compileGLSLStage, glslType } from "./backends/rmsl-glsl";
import {
  WGSL_HELPERS, WGSL_UNIFORM_BINDING, WGSL_UNIFORM_STRUCT, compileWGSLStage,
  isWgslTexture, wgslMemberType, wgslType, wgslUniformLayout,
} from "./backends/rmsl-wgsl";
import { CompileCtx, CompileFnOptions } from "./backends/shared";
import { Node, ShaderType, var_ } from "./rmsl-core";

export function compileFnBody(
  result: Node<ShaderType>,
  params: Array<{ name: string; type: ShaderType }>,
  name: string,
  language: "glsl" | "wgsl",
): string {
  if (Array.isArray(result)) {
    throw new Error(
      "compileGLSLFn/compileWGSLFn does not support multi-return functions. "
      + "Define separate functions for each return value.",
    );
  }

  if (language === "glsl") {
    const ctx: CompileCtx = {
      nextId: 0,
      shaderStage: "fragment",
      uniforms: new Map(),
      attributes: new Map(),
      varyings: new Map(),
      outputs: new Map(),
      wgslSamplers: new Map(),
      varDefs: new Map(),
      memo: new Map(),
      wgslHelpers: new Set(),
      positionWritten: false,
      inFn: false,
      fragDepthUsed: false,
    fragCoordUsed: false,
      jsParams: new Set(),
      jsHelpers: new Set(),
      outTarget: null,
      derivatives: "throw",
      reentrant: false,
      jsNeedsRes: false,
    };
    const compiled = compileGLSLStage(result, ctx);
    const returnType = glslType((result as any)._t || "float");
    const paramStr = params.map(p => `${glslType(p.type)} ${p.name}`).join(", ");
    let code = "";
    ctx.uniforms.forEach((info) => {
      code += info.length !== undefined
        ? `uniform ${info.type} ${info.slot}[${info.length}];\n`
        : `uniform ${info.type} ${info.slot};\n`;
    });
    if (ctx.uniforms.size > 0) {
      code += "\n";
    }
    code += `${returnType} ${name}(${paramStr}) {\n`;
    for (const line of compiled.body) {
      code += `  ${line}\n`;
    }
    if (compiled.expr !== "0.0") {
      code += `  return ${compiled.expr};\n`;
    } else {
      code += `  return ${returnType}(0);\n`;
    }
    code += `}`;
    return code;
  } else {
    const ctx: CompileCtx = {
      nextId: 0,
      shaderStage: "fragment",
      uniforms: new Map(),
      attributes: new Map(),
      varyings: new Map(),
      outputs: new Map(),
      wgslSamplers: new Map(),
      varDefs: new Map(),
      memo: new Map(),
      wgslHelpers: new Set(),
      positionWritten: false,
      inFn: false,
      fragDepthUsed: false,
    fragCoordUsed: false,
      jsParams: new Set(),
      jsHelpers: new Set(),
      outTarget: null,
      derivatives: "throw",
      reentrant: false,
      jsNeedsRes: false,
    };
    const compiled = compileWGSLStage(result, ctx);
    const returnType = wgslType((result as any)._t || "float");
    const paramStr = params.map(p => `${p.name}: ${wgslType(p.type)}`).join(", ");
    // Helpers standing in for GLSL builtins WGSL lacks, emitted ahead of the
    // function that calls them. The whole-shader path does the same at its own
    // top level; a function emitted on its own has to carry them itself, or it
    // calls something that was never defined. Sorted so identical input gives
    // identical output regardless of the order ops were reached.
    let code = "";
    for (const helper of [...ctx.wgslHelpers].sort()) {
      code += `${WGSL_HELPERS[helper]}\n\n`;
    }
    code += `fn ${name}(${paramStr}) -> ${returnType} {\n`;
    for (const line of compiled.decls) {
      code += `  ${line}\n`;
    }
    for (const line of compiled.body) {
      code += `  ${line}\n`;
    }
    if (compiled.expr !== "0.0") {
      code += `  return ${compiled.expr};\n`;
    } else {
      code += `  return ${returnType}();\n`;
    }
    code += `}\n`;
    // Same single-struct packing as a full shader, for the same reason: one
    // binding per uniform runs out at twelve.
    let sortedUniforms = [...ctx.uniforms.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    // A texture is sampled through a companion sampler, so both are declared
    // or neither resolves. The whole-shader path does the same, in the same
    // binding groups.
    let samplerDecls = "";
    let samplerBinding = 0;
    ctx.wgslSamplers.forEach((info) => {
      samplerDecls += `@group(2) @binding(${samplerBinding++}) var ${info.samplerSlot}: sampler;\n`;
    });
    let textureDecls = "";
    let texBinding = 0;
    for (let [, info] of sortedUniforms.filter(([, i]) => isWgslTexture(i.type))) {
      textureDecls += `@group(1) @binding(${texBinding++}) var ${info.slot}: ${info.type};\n`;
    }
    if (textureDecls || samplerDecls) code = textureDecls + samplerDecls + "\n" + code;
    let plainUniforms = sortedUniforms.filter(([, i]) => !isWgslTexture(i.type));
    if (plainUniforms.length > 0) {
      let layout = wgslUniformLayout(
        plainUniforms.map(([, i]) => ({ slot: i.slot, type: i.type, length: i.length })),
      );
      let struct = `struct ${WGSL_UNIFORM_STRUCT} {\n`
        + layout.members.map(m => `  ${m.name}: ${wgslMemberType(m)},\n`).join("")
        + `};\n`
        + `@group(0) @binding(0) var<uniform> ${WGSL_UNIFORM_BINDING}: ${WGSL_UNIFORM_STRUCT};\n\n`;
      code = struct + code;
    }
    return code;
  }
}

export function compileGLSLFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): string {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const result = fn(...paramNodes);
  return compileFnBody(result, options.params, options.name, "glsl");
}

export function compileWGSLFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): string {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const result = fn(...paramNodes);
  return compileFnBody(result, options.params, options.name, "wgsl");
}

