import {
  attribute,
  bool,
  Break,
  float,
  Fn,
  fragCoord,
  If,
  int,
  Node,
  output,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  While,
} from "@random-mesh/rmsl";

// Split constant for Dekker's split of f32: 2^13 + 1 = 8193.0
const SPLIT = float(8193.0);

/**
 * Double-Single (DS) Addition: (a.x + a.y) + (b.x + b.y)
 * Using Knuth's TwoSum algorithm translated from WASM demo
 */
export const ds_add = Fn((a: Node<"vec2">, b: Node<"vec2">): Node<"vec2"> => {
  const x = a.x.add(b.x).toVar();
  const bv = x.sub(a.x).toVar();
  const av = x.sub(bv).toVar();
  const br = b.x.sub(bv).toVar();
  const ar = a.x.sub(av).toVar();
  const y = ar.add(br).toVar();
  const lo = y.add(a.y).add(b.y).toVar();
  const t = x.add(lo).toVar();
  return vec2(t, x.sub(t).add(lo));
});

/**
 * Double-Single (DS) Subtraction: a - b
 */
export const ds_sub = Fn((a: Node<"vec2">, b: Node<"vec2">): Node<"vec2"> => {
  return ds_add(a, vec2(b.x.negate(), b.y.negate()));
});

/**
 * Double-Single (DS) Multiplication: (a.x + a.y) * (b.x + b.y)
 * Dekker's TwoProd algorithm adapted to 2x float32
 */
export const ds_mul = Fn((a: Node<"vec2">, b: Node<"vec2">): Node<"vec2"> => {
  const t1 = SPLIT.mul(a.x).toVar();
  const a1 = t1.sub(t1.sub(a.x)).toVar();
  const a0 = a.x.sub(a1).toVar();

  const t2 = SPLIT.mul(b.x).toVar();
  const b1 = t2.sub(t2.sub(b.x)).toVar();
  const b0 = b.x.sub(b1).toVar();

  const p_hi = a.x.mul(b.x).toVar();
  const p_lo = a1.mul(b1).sub(p_hi).add(a1.mul(b0)).add(a0.mul(b1)).add(a0.mul(b0)).toVar();
  const p_lo2 = p_lo.add(a.x.mul(b.y)).add(a.y.mul(b.x)).toVar();

  const h = p_hi.add(p_lo2).toVar();
  const dstLo = p_lo2.sub(h.sub(p_hi)).toVar();
  const dstHi = h;
  return vec2(dstHi, dstLo);
});

// Full-screen quad attributes & varyings
export const quadPos = attribute("vec2");
export const v_pos = varying("vec2");

export const vertexMain = Fn(() => {
  v_pos.assign(quadPos);
  return vec4(quadPos.x, quadPos.y, 0.0, 1.0);
});

// Uniform declarations
export const u_resolution = uniform("vec2");
export const u_maxIter = uniform("int");
export const u_useHighPrecision = uniform("int");
export const u_pan_hi = uniform("vec2");
export const u_pan_lo = uniform("vec2");
export const u_scale_hi = uniform("vec2");
export const u_scale_lo = uniform("vec2");
export const u_palette = uniform("int");
// CPU-only: how many rows above this call's own row 0 the caller's row range
// actually starts at — 0 for a normal single-call draw, or a worker's row
// offset when a worker-pool splits one frame's rows across several draw()
// calls into disjoint slices of one shared output buffer.
export const u_rowOffset = uniform("float");

/**
 * The colour at a pixel `(dx, dy)` pixels from the view's center, shared by
 * the GPU fragment stage (whose `dx`/`dy` come from an interpolated
 * varying) and the CPU targets' `.draw()` (whose `dx`/`dy` come from
 * `fragCoord()` directly, with no vertex/varying stage at all).
 */
export const mandelbrotColorAt = Fn((dx: Node<"float">, dy: Node<"float">): Node<"vec4"> => {
  const iter = int(0).toVar();
  const magSq = float(0.0).toVar();
  const escaped = bool(false).toVar();

  If(u_useHighPrecision.equal(int(1)), () => {
    const dx_ds = vec2(dx, 0.0);
    const dy_ds = vec2(dy, 0.0);

    const cx_ds = vec2(u_pan_hi.x, u_pan_lo.x);
    const cy_ds = vec2(u_pan_hi.y, u_pan_lo.y);
    const scaleX_ds = vec2(u_scale_hi.x, u_scale_lo.x);
    const scaleY_ds = vec2(u_scale_hi.y, u_scale_lo.y);

    const cx = ds_add(cx_ds, ds_mul(scaleX_ds, dx_ds)).toVar();
    const cy = ds_add(cy_ds, ds_mul(scaleY_ds, dy_ds)).toVar();

    const zx = vec2(0.0, 0.0).toVar();
    const zy = vec2(0.0, 0.0).toVar();

    While(iter.lessThan(u_maxIter), () => {
      const zx2 = ds_mul(zx, zx).toVar();
      const zy2 = ds_mul(zy, zy).toVar();
      const magDS = ds_add(zx2, zy2).toVar();

      If(magDS.x.greaterThan(4.0), () => {
        escaped.assign(bool(true));
        magSq.assign(magDS.x);
        Break();
      });

      const diff = ds_sub(zx2, zy2).toVar();
      const new_zx = ds_add(diff, cx).toVar();

      const prod = ds_mul(zx, zy).toVar();
      const prod2 = ds_add(prod, prod).toVar();
      const new_zy = ds_add(prod2, cy).toVar();

      zx.assign(new_zx);
      zy.assign(new_zy);
      iter.assign(iter.add(int(1)));
    });
  }).Else(() => {
    const cx_f = u_pan_hi.x.add(dx.mul(u_scale_hi.x)).toVar();
    const cy_f = u_pan_hi.y.add(dy.mul(u_scale_hi.y)).toVar();
    const zx_f = float(0.0).toVar();
    const zy_f = float(0.0).toVar();

    While(iter.lessThan(u_maxIter), () => {
      const zx2_f = zx_f.mul(zx_f).toVar();
      const zy2_f = zy_f.mul(zy_f).toVar();
      const mag_f = zx2_f.add(zy2_f).toVar();

      If(mag_f.greaterThan(4.0), () => {
        escaped.assign(bool(true));
        magSq.assign(mag_f);
        Break();
      });

      const new_zy_f = float(2.0).mul(zx_f).mul(zy_f).add(cy_f).toVar();
      const new_zx_f = zx2_f.sub(zy2_f).add(cx_f).toVar();

      zx_f.assign(new_zx_f);
      zy_f.assign(new_zy_f);
      iter.assign(iter.add(int(1)));
    });
  });

  const finalColor = vec4(0.0, 0.0, 0.0, 1.0).toVar();

  If(escaped, () => {
    const logMag = magSq.log().mul(0.5).toVar();
    const nu = logMag.log().div(float(Math.LN2)).toVar();
    const smoothIter = float(iter).add(float(1.0)).sub(nu).toVar();

    const t = smoothIter.mul(0.05).toVar();
    const color = vec3(0.0, 0.0, 0.0).toVar();

    If(u_palette.equal(int(0)), () => {
      const r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
      const g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(2.0943951).cos()));
      const b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.1887902).cos()));
      color.assign(vec3(r, g, b));
    })
      .ElseIf(u_palette.equal(int(1)), () => {
        const r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
        const g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.6).cos()));
        const b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(1.2).cos()));
        color.assign(vec3(r, g, b));
      })
      .ElseIf(u_palette.equal(int(2)), () => {
        const r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(3.0).cos()));
        const g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.0).cos()));
        const b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(1.0).cos()));
        color.assign(vec3(r, g, b));
      })
      .Else(() => {
        const r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
        const g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(2.0).cos()));
        const b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.0).cos()));
        color.assign(vec3(r, g, b));
      });

    finalColor.assign(vec4(color, 1.0));
  }).Else(() => {
    finalColor.assign(vec4(0.0, 0.0, 0.0, 1.0));
  });

  return finalColor;
});

/** The GPU fragment stage: `dx`/`dy` come from the rasterizer-interpolated `v_pos`. */
export const calcMandelbrot = Fn(() => {
  const outColor = output("vec4");

  // Offset in pixels relative to center
  const dx = v_pos.x.mul(0.5).mul(u_resolution.x).toVar();
  const dy = v_pos.y.mul(0.5).mul(u_resolution.y).toVar();

  outColor.assign(mandelbrotColorAt(dx, dy));
  return outColor;
});

/**
 * The CPU-target entry point: no vertex stage, no varying — `.draw()` feeds
 * each pixel's center in as `fragCoord()` directly. `fragCoord().y` grows
 * downward (row 0 is the top row, matching `ImageData`'s layout), the
 * opposite of `v_pos.y` (which grows upward, matching GL clip space), so the
 * two convert to the same pixel-offset convention with a subtraction rather
 * than a multiply.
 */
export const calcMandelbrotCpu = Fn(() => {
  const dx = fragCoord().x.sub(u_resolution.x.mul(0.5)).toVar();
  const dy = u_resolution.y.mul(0.5).sub(fragCoord().y.add(u_rowOffset)).toVar();
  return mandelbrotColorAt(dx, dy);
});
