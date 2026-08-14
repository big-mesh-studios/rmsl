import {
  float, int, vec2, vec3, vec4, bool,
  Fn, If, While, Break,
  attribute, uniform, varying, output,
  Node,
} from "@random-mesh/rmsl";

// Split constant for Dekker's split of f32: 2^13 + 1 = 8193.0
const SPLIT = float(8193.0);

/**
 * Double-Single (DS) Addition: (a.x + a.y) + (b.x + b.y)
 * Using Knuth's TwoSum algorithm translated from WASM demo
 */
export const ds_add = Fn((a: Node<"vec2">, b: Node<"vec2">) => {
  let x = a.x.add(b.x).toVar();
  let bv = x.sub(a.x).toVar();
  let av = x.sub(bv).toVar();
  let br = b.x.sub(bv).toVar();
  let ar = a.x.sub(av).toVar();
  let y = ar.add(br).toVar();
  let lo = y.add(a.y).add(b.y).toVar();
  let t = x.add(lo).toVar();
  return vec2(t, x.sub(t).add(lo));
});

/**
 * Double-Single (DS) Subtraction: a - b
 */
export const ds_sub = Fn((a: Node<"vec2">, b: Node<"vec2">) => {
  return ds_add(a, vec2(b.x.negate(), b.y.negate()));
});

/**
 * Double-Single (DS) Multiplication: (a.x + a.y) * (b.x + b.y)
 * Dekker's TwoProd algorithm adapted to 2x float32
 */
export const ds_mul = Fn((a: Node<"vec2">, b: Node<"vec2">) => {
  let t1 = SPLIT.mul(a.x).toVar();
  let a1 = t1.sub(t1.sub(a.x)).toVar();
  let a0 = a.x.sub(a1).toVar();

  let t2 = SPLIT.mul(b.x).toVar();
  let b1 = t2.sub(t2.sub(b.x)).toVar();
  let b0 = b.x.sub(b1).toVar();

  let p_hi = a.x.mul(b.x).toVar();
  let p_lo = a1.mul(b1).sub(p_hi).add(a1.mul(b0)).add(a0.mul(b1)).add(a0.mul(b0)).toVar();
  let p_lo2 = p_lo.add(a.x.mul(b.y)).add(a.y.mul(b.x)).toVar();

  let h = p_hi.add(p_lo2).toVar();
  let dstLo = p_lo2.sub(h.sub(p_hi)).toVar();
  let dstHi = h;
  return vec2(dstHi, dstLo);
});

// Full-screen quad attributes & varyings
export let quadPos = attribute("vec2");
export let v_pos = varying("vec2");

export let vertexMain = Fn(() => {
  v_pos.assign(quadPos);
  return vec4(quadPos.x, quadPos.y, 0.0, 1.0);
});

// Uniform declarations
export let u_resolution = uniform("vec2");
export let u_maxIter = uniform("int");
export let u_useHighPrecision = uniform("int");
export let u_pan_hi = uniform("vec2");
export let u_pan_lo = uniform("vec2");
export let u_scale_hi = uniform("vec2");
export let u_scale_lo = uniform("vec2");
export let u_palette = uniform("int");

export let calcMandelbrot = Fn(() => {
  let outColor = output("vec4");

  // Offset in pixels relative to center
  let dx = v_pos.x.mul(0.5).mul(u_resolution.x).toVar();
  let dy = v_pos.y.mul(0.5).mul(u_resolution.y).toVar();

  let iter = int(0).toVar();
  let magSq = float(0.0).toVar();
  let escaped = bool(false).toVar();

  If(u_useHighPrecision.equal(int(1)), () => {
    let dx_ds = vec2(dx, 0.0);
    let dy_ds = vec2(dy, 0.0);

    let cx_ds = vec2(u_pan_hi.x, u_pan_lo.x);
    let cy_ds = vec2(u_pan_hi.y, u_pan_lo.y);
    let scaleX_ds = vec2(u_scale_hi.x, u_scale_lo.x);
    let scaleY_ds = vec2(u_scale_hi.y, u_scale_lo.y);

    let cx = ds_add(cx_ds, ds_mul(scaleX_ds, dx_ds)).toVar();
    let cy = ds_add(cy_ds, ds_mul(scaleY_ds, dy_ds)).toVar();

    let zx = vec2(0.0, 0.0).toVar();
    let zy = vec2(0.0, 0.0).toVar();

    While(iter.lessThan(u_maxIter), () => {
      let zx2 = ds_mul(zx, zx).toVar();
      let zy2 = ds_mul(zy, zy).toVar();
      let magDS = ds_add(zx2, zy2).toVar();

      If(magDS.x.greaterThan(4.0), () => {
        escaped.assign(bool(true));
        magSq.assign(magDS.x);
        Break();
      });

      let diff = ds_sub(zx2, zy2).toVar();
      let new_zx = ds_add(diff, cx).toVar();

      let prod = ds_mul(zx, zy).toVar();
      let prod2 = ds_add(prod, prod).toVar();
      let new_zy = ds_add(prod2, cy).toVar();

      zx.assign(new_zx);
      zy.assign(new_zy);
      iter.assign(iter.add(int(1)));
    });
  }).Else(() => {
    let cx_f = u_pan_hi.x.add(dx.mul(u_scale_hi.x)).toVar();
    let cy_f = u_pan_hi.y.add(dy.mul(u_scale_hi.y)).toVar();
    let zx_f = float(0.0).toVar();
    let zy_f = float(0.0).toVar();

    While(iter.lessThan(u_maxIter), () => {
      let zx2_f = zx_f.mul(zx_f).toVar();
      let zy2_f = zy_f.mul(zy_f).toVar();
      let mag_f = zx2_f.add(zy2_f).toVar();

      If(mag_f.greaterThan(4.0), () => {
        escaped.assign(bool(true));
        magSq.assign(mag_f);
        Break();
      });

      let new_zy_f = float(2.0).mul(zx_f).mul(zy_f).add(cy_f).toVar();
      let new_zx_f = zx2_f.sub(zy2_f).add(cx_f).toVar();

      zx_f.assign(new_zx_f);
      zy_f.assign(new_zy_f);
      iter.assign(iter.add(int(1)));
    });
  });

  If(escaped, () => {
    let logMag = magSq.log().mul(0.5).toVar();
    let nu = logMag.log().div(float(Math.LN2)).toVar();
    let smoothIter = float(iter).add(float(1.0)).sub(nu).toVar();

    let t = smoothIter.mul(0.05).toVar();
    let color = vec3(0.0, 0.0, 0.0).toVar();

    If(u_palette.equal(int(0)), () => {
      let r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
      let g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(2.0943951).cos()));
      let b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.1887902).cos()));
      color.assign(vec3(r, g, b));
    }).ElseIf(u_palette.equal(int(1)), () => {
      let r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
      let g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.6).cos()));
      let b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(1.2).cos()));
      color.assign(vec3(r, g, b));
    }).ElseIf(u_palette.equal(int(2)), () => {
      let r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(3.0).cos()));
      let g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.0).cos()));
      let b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(1.0).cos()));
      color.assign(vec3(r, g, b));
    }).Else(() => {
      let r = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(0.0).cos()));
      let g = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(2.0).cos()));
      let b = float(0.5).add(float(0.5).mul(t.mul(6.2831853).add(4.0).cos()));
      color.assign(vec3(r, g, b));
    });

    outColor.assign(vec4(color, 1.0));
  }).Else(() => {
    outColor.assign(vec4(0.0, 0.0, 0.0, 1.0));
  });

  return outColor;
});
