import {
  float, Fn, If, uniform, varying, output,
  attribute, bool, Node, vec2, vec3, vec4, builtinFragDepth,
} from "@random-mesh/rmsl";

// === I/O declarations ===
export let quadPos = attribute("vec2");
export let positionWorld = varying("vec3");
export let positionGeometry = varying("vec3");
export let cameraProjectionMatrix = uniform("mat4");
export let cameraViewMatrix = uniform("mat4");
export let cameraProjectionMatrixInverse = uniform("mat4");
export let cameraWorldMatrix = uniform("mat4");
export let cameraPosition = uniform("vec3");

// === Vertex shader — full-screen quad ===
export let vertexMain = Fn(() => {
  positionGeometry.assign(vec3(quadPos.x, quadPos.y, 0.0));
  positionWorld.assign(vec3(quadPos.x, 0.0, quadPos.y));
  return vec4(quadPos.x, quadPos.y, 0.0, 1.0);
});

// === Fragment shader helpers ===
export let getGrid = Fn((size: Node<"float">, p: Node<"vec3">, fw: Node<"vec2">) => {
  let r = p.xz.div(size).toVar();
  let grid = r.sub(0.5).fract().sub(0.5).abs().div(fw.div(size));
  let line = grid.x.min(grid.y);
  return float(1.0).sub(line.mul(0.5).min(1.0));
});

export let calcColourAndDepth = Fn(() => {
  let isOrthographic = bool(false).toVar();

  let skyColour = vec3(0.3, 0.4, 0.6);
  let horizSkyColour = vec3(0.5, 0.5, 0.6);

  let ro = vec3().toVar();
  let rd = vec3().toVar();

  If(isOrthographic, () => {
    rd.assign(vec3(
      cameraViewMatrix.element(0).z,
      cameraViewMatrix.element(1).z,
      cameraViewMatrix.element(2).z,
    ).negate().normalize());
    ro.assign(positionWorld);
  }).Else(() => {
    ro.assign(cameraPosition);
    let viewPos = cameraProjectionMatrixInverse.mul(vec4(positionGeometry.x, positionGeometry.y, -1.0, 1.0));
    let worldDir = cameraWorldMatrix.mul(vec4(viewPos.x, viewPos.y, viewPos.z, 0.0));
    rd.assign(worldDir.xyz.normalize());
  });

  // Compute ground intersection and its screen-space derivative (uniform control flow)
  let p = ro.add(rd.mul(ro.y.negate().div(rd.y))).toVar();
  let pFWidth = p.xz.fwidth().toVar();

  let colour = vec3(0.7, 0.7, 0.7).toVar();
  let outColor = output("vec4");
  let fragDepth = builtinFragDepth();

  let groundColour = vec3(0.7, 0.7, 0.7).toVar();
  let gridColour = vec3(0.5, 0.5, 0.5).toVar();
  If(isOrthographic, () => {
    groundColour.assign(vec3(0.9, 0.9, 0.9));
    gridColour.assign(vec3(0.8, 0.8, 0.8));
  });

  let isPerspective = isOrthographic.not();

  If(isPerspective.and(ro.y.lessThan(0.0)), () => {
    colour.assign(skyColour);
  }).ElseIf(isPerspective.and(rd.y.greaterThan(0.3)), () => {
    colour.assign(skyColour);
  }).ElseIf(isPerspective.and(rd.y.greaterThan(0.0)), () => {
    let t = rd.y.div(0.03).clamp(0, 1);
    colour.assign(horizSkyColour.mix(skyColour, t));
  }).Else(() => {
    let fadeFactor = float(1.0).toVar();
    If(isOrthographic, () => {
      fadeFactor.assign(rd.y.abs());
    }).Else(() => {
      fadeFactor.assign(float(1.0).sub(ro.y.div(float(8000.0))).clamp(0.0, 1.0));
      fadeFactor.assign(fadeFactor.pow(3.0));
    });
    If(rd.y.abs().lessThan(0.0001), () => {
      colour.assign(groundColour);
    }).Else(() => {
      let refDist = float(1.0).toVar();
      If(isOrthographic, () => {
        refDist.assign(float(1.0).div(cameraProjectionMatrix.element(0).x.abs().max(float(0.001))));
      }).Else(() => {
        refDist.assign(cameraPosition.y.abs().mul(0.1).max(float(0.001)));
      });
      let exponent = refDist.log().div(float(Math.log(10))).floor().clamp(-3, 6);
      let minorSize = float(10.0).pow(exponent);
      let g1 = getGrid(minorSize, p, pFWidth).toVar();
      let g2 = getGrid(minorSize.mul(10.0), p, pFWidth).toVar();
      let fc = vec4(1.0, 1.0, 1.0, g2.mix(g1, g1).mul(fadeFactor)).toVar();
      let fca = fc.a.mul(0.5).mix(fc.a, g2);
      If(fca.lessThanEqual(0.0), () => {
        colour.assign(groundColour);
      }).Else(() => {
        colour.assign(groundColour.mix(gridColour, fca));
      });
    });
  });

  If(rd.y.lessThan(-0.001), () => {
    let t = ro.y.negate().div(rd.y);
    let p = ro.add(rd.mul(t));
    let clipPos = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(p.x, p.y, p.z, 1.0)).toVar();
    let ndcZ = clipPos.z.div(clipPos.w);
    fragDepth.assign(ndcZ.mul(0.5).add(0.5));
  });

  outColor.assign(vec4(colour, 1.0));
  return outColor;
});

// === Matrix math utilities (column-major Float32Array) ===
export function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array<ArrayBuffer> {
  let f = 1 / Math.tan(fovY / 2);
  let nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function mat4LookAt(eyeX: number, eyeY: number, eyeZ: number, cx: number, cy: number, cz: number, upX: number, upY: number, upZ: number): Float32Array<ArrayBuffer> {
  let zx = eyeX - cx, zy = eyeY - cy, zz = eyeZ - cz;
  let zl = Math.sqrt(zx * zx + zy * zy + zz * zz);
  zx /= zl; zy /= zl; zz /= zl;
  let xx = upY * zz - upZ * zy;
  let xy = upZ * zx - upX * zz;
  let xz = upX * zy - upY * zx;
  let xl = Math.sqrt(xx * xx + xy * xy + xz * xz);
  xx /= xl; xy /= xl; xz /= xl;
  let yx = zy * xz - zz * xy;
  let yy = zz * xx - zx * xz;
  let yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eyeX + xy * eyeY + xz * eyeZ),
    -(yx * eyeX + yy * eyeY + yz * eyeZ),
    -(zx * eyeX + zy * eyeY + zz * eyeZ),
    1,
  ]);
}

export function mat4Inverse(m: Float32Array): Float32Array<ArrayBuffer> {
  let a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  let a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  let a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  let a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  let b00 = a00 * a11 - a01 * a10;
  let b01 = a00 * a12 - a02 * a10;
  let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11;
  let b04 = a01 * a13 - a03 * a11;
  let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30;
  let b07 = a20 * a32 - a22 * a30;
  let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31;
  let b10 = a21 * a33 - a23 * a31;
  let b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return new Float32Array(16);
  let id = 1 / det;
  let out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * id;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * id;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * id;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * id;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * id;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * id;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * id;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * id;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * id;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * id;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * id;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * id;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * id;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * id;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * id;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * id;
  return out;
}

// === Quad vertices ===
export let quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
