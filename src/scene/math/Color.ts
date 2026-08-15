/**
 * An sRGB color, components in 0..1, following three.js's `Color`.
 */
export class Color {
  r: number;
  g: number;
  b: number;

  constructor(r = 1, g = 1, b = 1) {
    this.r = r;
    this.g = g;
    this.b = b;
  }

  set(r: number, g: number, b: number): this {
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }

  copy(color: Color): this {
    this.r = color.r;
    this.g = color.g;
    this.b = color.b;
    return this;
  }

  clone(): Color {
    return new Color(this.r, this.g, this.b);
  }

  setRGB(r: number, g: number, b: number): this {
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }

  setHex(hex: number): this {
    hex = Math.floor(hex);
    this.r = (hex >> 16 & 255) / 255;
    this.g = (hex >> 8 & 255) / 255;
    this.b = (hex & 255) / 255;
    return this;
  }

  getHex(): number {
    return (Math.round(this.r * 255) << 16) | (Math.round(this.g * 255) << 8) | Math.round(this.b * 255);
  }

  setStyle(style: string): this {
    if (/^#([0-9a-fA-F]{6})$/.test(style)) {
      return this.setHex(parseInt(style.slice(1), 16));
    }
    if (/^#([0-9a-fA-F]{3})$/.test(style)) {
      const m = style.slice(1);
      return this.setHex(parseInt(m[0] + m[0] + m[1] + m[1] + m[2] + m[2], 16));
    }
    if (/^(rgb|rgba)\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.test(style)) {
      const m = /^rgb\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(style)!;
      return this.setRGB(+m[1] / 255, +m[2] / 255, +m[3] / 255);
    }
    throw new Error(`unsupported color style: ${style}`);
  }

  multiplyScalar(scalar: number): this {
    this.r *= scalar;
    this.g *= scalar;
    this.b *= scalar;
    return this;
  }

  multiply(color: Color): this {
    this.r *= color.r;
    this.g *= color.g;
    this.b *= color.b;
    return this;
  }

  lerp(color: Color, alpha: number): this {
    this.r += (color.r - this.r) * alpha;
    this.g += (color.g - this.g) * alpha;
    this.b += (color.b - this.b) * alpha;
    return this;
  }

  equals(color: Color): boolean {
    return color.r === this.r && color.g === this.g && color.b === this.b;
  }

  toArray(array: number[] = [], offset = 0): number[] {
    array[offset] = this.r;
    array[offset + 1] = this.g;
    array[offset + 2] = this.b;
    return array;
  }
}
