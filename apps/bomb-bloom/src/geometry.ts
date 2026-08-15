// Indexed-mesh geometry generated on the CPU: the bomb body (sphere, cylinder
// cap, wick tube) and the flame particle billboards. Flame particles mirror
// melty-karts' `models/Bomb.tsx` `_sharedWickFire`.

export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array;
}

export interface FlameGeometry {
  positions: Float32Array; // base particle positions
  corners: Float32Array;   // billboard corner (-1..1)
  drifts: Float32Array;
  lives: Float32Array;
  offsets: Float32Array;
  sizes: Float32Array;
  spins: Float32Array;
  uvs: Float32Array;
  indices: Uint16Array;
}

// === Catmull-Rom ===

function vec3Sub(a: number[], b: number[]): number[] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Cross(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vec3Normalize(v: number[]): number[] {
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * Sample a Catmull-Rom spline through `points`. For each segment the four
 * control points are the neighbours, clamped at the ends so the curve runs
 * through every point.
 */
export function catmullRom(points: number[][], samplesPerSegment: number): number[][] {
  const out: number[][] = [];
  const n = points.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(i + 2, n - 1)];
    for (let s = 0; s < samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      const t2 = t * t;
      const t3 = t2 * t;
      out.push([
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        0.5 * ((2 * p1[2]) + (-p0[2] + p2[2]) * t + (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 + (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3),
      ]);
    }
  }
  out.push([...points[n - 1]]);
  return out;
}

/** A tube (swept circle) along a sampled curve. */
export function buildTube(curve: number[][], radius: number, radialSegments: number): Mesh {
  const n = curve.length;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < n; i++) {
    const p = curve[i];
    const prev = curve[Math.max(i - 1, 0)];
    const next = curve[Math.min(i + 1, n - 1)];
    let t = vec3Normalize(vec3Sub(next, prev));
    if (t[0] === 0 && t[1] === 0 && t[2] === 0) t = [0, 1, 0];
    // Pick an up vector not parallel to the tangent.
    const up = Math.abs(t[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const binormal = vec3Normalize(vec3Cross(t, up));
    const normal = vec3Cross(binormal, t);
    for (let s = 0; s < radialSegments; s++) {
      const a = (s / radialSegments) * Math.PI * 2;
      const nx = Math.cos(a), ny = Math.sin(a);
      positions.push(
        p[0] + radius * (nx * normal[0] + ny * binormal[0]),
        p[1] + radius * (nx * normal[1] + ny * binormal[1]),
        p[2] + radius * (nx * normal[2] + ny * binormal[2]),
      );
      normals.push(
        nx * normal[0] + ny * binormal[0],
        nx * normal[1] + ny * binormal[1],
        nx * normal[2] + ny * binormal[2],
      );
    }
  }

  for (let i = 0; i < n - 1; i++) {
    for (let s = 0; s < radialSegments; s++) {
      const a = i * radialSegments + s;
      const b = a + radialSegments;
      const a2 = i * radialSegments + ((s + 1) % radialSegments);
      const b2 = a2 + radialSegments;
      indices.push(a, b, b2, a, b2, a2);
    }
  }

  return toMesh(positions, normals, indices);
}

/** A UV-sphere centered at the origin. */
export function buildSphere(r: number, latSegments: number, longSegments: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= latSegments; i++) {
    const v = i / latSegments;
    const phi = v * Math.PI;
    const y = Math.cos(phi);
    const rr = Math.sin(phi);
    for (let j = 0; j <= longSegments; j++) {
      const u = j / longSegments;
      const theta = u * Math.PI * 2;
      const x = Math.cos(theta) * rr;
      const z = Math.sin(theta) * rr;
      positions.push(x * r, y * r, z * r);
      normals.push(x, y, z);
    }
  }
  for (let i = 0; i < latSegments; i++) {
    for (let j = 0; j < longSegments; j++) {
      const a = i * (longSegments + 1) + j;
      const b = a + longSegments + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return toMesh(positions, normals, indices);
}

/** A cylinder from y=0 to y=height, with side and caps. */
export function buildCylinder(r: number, height: number, segments: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const x = Math.cos(a), z = Math.sin(a);
    positions.push(x * r, 0, z * r, x * r, height, z * r);
    normals.push(x, 0, z, x, 0, z);
  }
  for (let i = 0; i < segments; i++) {
    const a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3;
    indices.push(a, b, c, b, d, c);
  }
  // Top and bottom caps.
  const topIdx = positions.length / 3;
  const bottomIdx = topIdx + 1;
  positions.push(0, height, 0, 0, 0, 0);
  normals.push(0, 1, 0, 0, -1, 0);
  for (let i = 0; i < segments; i++) {
    const a = i * 2, c = i * 2 + 2;
    indices.push(topIdx, c, a);
    indices.push(bottomIdx, a, c);
  }
  return toMesh(positions, normals, indices);
}

// === Flame particles ===

export function buildFlameParticles(): FlameGeometry {
  const particleCount = 48;
  const positions = new Float32Array(particleCount * 3);
  const drift = new Float32Array(particleCount * 3);
  const life = new Float32Array(particleCount);
  const offset = new Float32Array(particleCount);
  const size = new Float32Array(particleCount);
  const spin = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    const s = i * 3;
    positions[s + 0] = (Math.random() - 0.5) * 0.012;
    positions[s + 1] = Math.random() * 0.01;
    positions[s + 2] = (Math.random() - 0.5) * 0.012;
    drift[s + 0] = (Math.random() - 0.5) * 0.05;
    drift[s + 1] = 0.08 + Math.random() * 0.06;
    drift[s + 2] = (Math.random() - 0.5) * 0.05;
    life[i] = 0.45 + Math.random() * 0.35;
    offset[i] = Math.random() * life[i];
    size[i] = 16.0 + Math.random() * 20.0;
    spin[i] = Math.random() * Math.PI * 2.0;
  }

  const quad = particleCount * 4;
  const quadPositions = new Float32Array(quad * 3);
  const quadCorners = new Float32Array(quad * 2);
  const quadUvs = new Float32Array(quad * 2);
  const quadDrift = new Float32Array(quad * 3);
  const quadLife = new Float32Array(quad);
  const quadOffset = new Float32Array(quad);
  const quadSize = new Float32Array(quad);
  const quadSpin = new Float32Array(quad);
  const quadIndices = new Uint16Array(particleCount * 6);

  const cornerData = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

  for (let i = 0; i < particleCount; i++) {
    const v0 = i * 4;
    const ps = i * 3;
    for (let j = 0; j < 4; j++) {
      const vi = v0 + j;
      const vs = vi * 3;
      quadPositions[vs + 0] = positions[ps + 0];
      quadPositions[vs + 1] = positions[ps + 1];
      quadPositions[vs + 2] = positions[ps + 2];
      quadCorners[vi * 2 + 0] = cornerData[j][0];
      quadCorners[vi * 2 + 1] = cornerData[j][1];
      quadUvs[vi * 2 + 0] = (cornerData[j][0] + 1) / 2;
      quadUvs[vi * 2 + 1] = (cornerData[j][1] + 1) / 2;
      quadDrift[vs + 0] = drift[ps + 0];
      quadDrift[vs + 1] = drift[ps + 1];
      quadDrift[vs + 2] = drift[ps + 2];
      quadLife[vi] = life[i];
      quadOffset[vi] = offset[i];
      quadSize[vi] = size[i];
      quadSpin[vi] = spin[i];
    }
    quadIndices[i * 6 + 0] = v0 + 0;
    quadIndices[i * 6 + 1] = v0 + 1;
    quadIndices[i * 6 + 2] = v0 + 2;
    quadIndices[i * 6 + 3] = v0 + 0;
    quadIndices[i * 6 + 4] = v0 + 2;
    quadIndices[i * 6 + 5] = v0 + 3;
  }

  return {
    positions: quadPositions,
    corners: quadCorners,
    drifts: quadDrift,
    lives: quadLife,
    offsets: quadOffset,
    sizes: quadSize,
    spins: quadSpin,
    uvs: quadUvs,
    indices: quadIndices,
  };
}

function toMesh(positions: number[], normals: number[], indices: number[]): Mesh {
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint16Array(indices),
  };
}
