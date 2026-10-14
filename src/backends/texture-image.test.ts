import { type BackendOrigin, UsageError } from "../errors";
import { describe, expect, it } from "vitest";
import { textureImage } from "./texture-image";

const TEST_ORIGIN: BackendOrigin = { target: "glsl", caller: "textureImage" };

describe("textureImage", () => {
  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("expands a texture that stores fewer channels the way a device reads it", () => {
    const image = textureImage(
      TEST_ORIGIN,
      { data: Uint8Array.of(10, 20), width: 2, height: 1, channels: 1 },
      "sampler2D",
    );
    expect(Array.from(image.texels)).toEqual([10, 0, 0, 255, 20, 0, 0, 255]);
    expect(image.normalized).toBe(true);
  });

  /**
   * @canon spec-an-adapter-writes-a-texture-of-the-same-shape-in-place
   */
  it("hands four-channel data of the right array type on without copying it", () => {
    const data = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
    const image = textureImage(TEST_ORIGIN, { data, width: 2, height: 1 }, "sampler2D");
    expect(image.texels.buffer).toBe(data.buffer);
    expect(Array.from(image.texels)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const clamped = Uint8ClampedArray.of(9, 9, 9, 9);
    expect(textureImage(TEST_ORIGIN, { data: clamped, width: 1, height: 1 }, "sampler2D").texels.buffer).toBe(
      clamped.buffer,
    );
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("keeps the width of the integers an integer sampler reads", () => {
    const image = textureImage(TEST_ORIGIN, { data: Int16Array.of(-1, 2, 3, 4), width: 1, height: 1 }, "isampler2D");
    expect(image).toMatchObject({ bits: 16, signed: true, normalized: false });
    expect(Array.from(image.texels)).toEqual([-1, 2, 3, 4]);
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("refuses data a sampler of that type cannot read on a GPU target", () => {
    expect(() =>
      textureImage(TEST_ORIGIN, { data: Float32Array.of(1, 1, 1, 1), width: 1, height: 1 }, "sampler2D"),
    ).toThrowRmsl(UsageError, "texture-data-of-another-kind");
    expect(() =>
      textureImage(TEST_ORIGIN, { data: Uint8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "isampler2D"),
    ).toThrowRmsl(UsageError, "texture-data-of-another-kind");
    expect(() =>
      textureImage(TEST_ORIGIN, { data: Int8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "usampler2D"),
    ).toThrowRmsl(UsageError, "texture-data-of-another-kind");
  });

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("refuses data shorter than the texture and a cube sampler", () => {
    expect(() =>
      textureImage(TEST_ORIGIN, { data: Uint8Array.of(1, 1, 1, 1), width: 2, height: 2 }, "sampler2D"),
    ).toThrowRmsl(UsageError, "texture-data-size");
    expect(() =>
      textureImage(TEST_ORIGIN, { data: Uint8Array.of(1, 1, 1, 1), width: 1, height: 1 }, "samplerCube"),
    ).toThrowRmsl(UsageError, "texture-kind-unsupported");
  });
});
