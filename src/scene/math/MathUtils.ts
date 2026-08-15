export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function euclideanModulo(n: number, m: number): number {
  return ((n % m) + m) % m;
}

export function lerp(x: number, y: number, t: number): number {
  return x + (y - x) * t;
}

export function inverseLerp(x: number, y: number, value: number): number {
  if (x !== y) return clamp((value - x) / (y - x), 0, 1);
  return 0;
}

export function mapLinear(x: number, a1: number, a2: number, b1: number, b2: number): number {
  return b1 + (x - a1) * (b2 - b1) / (a2 - a1);
}

export function smoothstep(x: number, min: number, max: number): number {
  if (x <= min) return 0;
  if (x >= max) return 1;
  x = (x - min) / (max - min);
  return x * x * (3 - 2 * x);
}

export function smootherstep(x: number, min: number, max: number): number {
  if (x <= min) return 0;
  if (x >= max) return 1;
  x = (x - min) / (max - min);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export function degToRad(degrees: number): number {
  return degrees * Math.PI / 180;
}

export function radToDeg(radians: number): number {
  return radians * 180 / Math.PI;
}

export function damp(x: number, y: number, lambda: number, dt: number): number {
  return lerp(x, y, 1 - Math.exp(-lambda * dt));
}

export function randFloat(low: number, high: number): number {
  return low + Math.random() * (high - low);
}

export function randInt(low: number, high: number): number {
  return low + Math.floor(Math.random() * (high - low + 1));
}

export function pingpong(x: number, length = 1): number {
  return length - Math.abs(euclideanModulo(x, length * 2) - length);
}

let _seed = 1234567;
export function seededRandom(s: number): number {
  if (s !== undefined) _seed = s % 2147483647;
  _seed = _seed * 16807 % 2147483647;
  return (_seed - 1) / 2147483646;
}

export function ceilPowerOfTwo(value: number): number {
  return Math.pow(2, Math.ceil(Math.log(value) / Math.LN2));
}

export function floorPowerOfTwo(value: number): number {
  return Math.pow(2, Math.floor(Math.log(value) / Math.LN2));
}

export function isPowerOfTwo(value: number): boolean {
  return (value & (value - 1)) === 0 && value !== 0;
}
