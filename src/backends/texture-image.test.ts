import { describe, expect, it } from "vitest";
import { textureImage } from "./texture-image";

describe("textureImage", () => {
  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("expands a texture that stores fewer channels the way a device reads it", () => {
    const image = textureImage({ data: Uint8Array.of(10, 20), width: 2, height: 1, channels: 1 }, "sampler2D");
    expect(Array.from(image.texels)).toEqual([10, 0, 0, 255, 20, 0, 0, 255]);
    expect(image.normalized).toBe(true);
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("keeps the width of the integers an integer sampler reads", () => {
    const image = textureImage({ data: Int16Array.of(-1, 2, 3, 4), width: 1, height: 1 }, "isampler2D");
    expect(image).toMatchObject({ bits: 16, signed: true, normalized: false });
    expect(Array.from(image.texels)).toEqual([-1, 2, 3, 4]);
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("refuses data a sampler of that type cannot read on a GPU target", () => {
    expect(() => textureImage({ data: Float32Array.of(1, 1, 1, 1), width: 1, height: 1 }, "sampler2D")).toThrow(
      /Uint8Array/,
    );
    expect(() => textureImage({ data: Uint8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "isampler2D")).toThrow(
      /Int8Array/,
    );
    expect(() => textureImage({ data: Int8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "usampler2D")).toThrow(
      /Uint8Array/,
    );
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("refuses data shorter than the texture and a cube sampler", () => {
    expect(() => textureImage({ data: Uint8Array.of(1, 1, 1, 1), width: 2, height: 2 }, "sampler2D")).toThrow(
      /needs 16 values/,
    );
    expect(() => textureImage({ data: Uint8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "samplerCube")).toThrow(
      /not supported/,
    );
  });
});
