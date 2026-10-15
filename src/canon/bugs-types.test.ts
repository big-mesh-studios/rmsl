import { describe, expect, it } from "vitest";
import ts from "typescript";

/**
 * The type errors the TypeScript checker reports for `source`, a module
 * beside this file, under the project's compiler options. A type that has to
 * change is pinned here, at run time, where a `.test-d.ts` file could only
 * fail the whole type check.
 */
function typeErrors(source: string): string[] {
  const file = new URL("./__bool-attribute-types.ts", import.meta.url).pathname;
  const options: ts.CompilerOptions = {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    strict: true,
    esModuleInterop: true,
    skipLibCheck: true,
    noEmit: true,
  };
  const host = ts.createCompilerHost(options);
  const read = host.getSourceFile.bind(host);
  host.getSourceFile = (name, version, ...rest) =>
    name === file ? ts.createSourceFile(name, source, version) : read(name, version, ...rest);
  const exists = host.fileExists.bind(host);
  host.fileExists = (name) => name === file || exists(name);
  const program = ts.createProgram([file], options, host);
  return ts
    .getPreEmitDiagnostics(program, program.getSourceFile(file))
    .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
}

describe("the type of setAttribute for a bool attribute", () => {
  /**
   * @canon spec-the-type-of-set-attribute-takes-an-int32-or-uint32-array-for-a-bool-attribute
   * @canon bug-the-type-of-set-attribute-takes-a-float-array-for-a-bool-attribute
   */
  it.fails(
    "takes an Int32Array or a Uint32Array, and no float array, on every adapter",
    () => {
      const adapters = ["GlslAdapter", "WgslAdapter", "JsAdapter", "WasmAdapter"];
      const call = (array: string) =>
        adapters
          .map(
            (adapter) =>
              `declare const a${adapter}: ${adapter};\na${adapter}.setAttribute(flag, new ${array}(3));\na${adapter}.setAttribute(flags, new ${array}(9));`,
          )
          .join("\n");
      const header = `import { attribute } from "../rmsl";
import type { GlslAdapter } from "../glsl";
import type { WgslAdapter } from "../wgsl";
import type { JsAdapter } from "../js";
import type { WasmAdapter } from "../wasm";
const flag = attribute("bool");
const flags = attribute("bvec3");
`;
      expect(typeErrors(header + call("Int32Array"))).toEqual([]);
      expect(typeErrors(header + call("Uint32Array"))).toEqual([]);
      expect(typeErrors(header + call("Float32Array"))).toHaveLength(8);
    },
    60_000,
  );
});
