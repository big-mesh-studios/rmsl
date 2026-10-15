/**
 * Raw WebGPU probe: which vertex supplies the flat (default `first` sampling)
 * varying of each triangle in a triangle-strip draw. Runs in Deno and in a browser.
 * Returns a plain JSON-serialisable result object.
 */
export async function runProbe(gpu) {
  const out = { errors: [] };
  const stage = (n) => { globalThis.__probeStage = n; };
  stage("requestAdapter");
  const adapter = await gpu.requestAdapter();
  stage("requestDevice");
  if (!adapter) return { error: "no adapter" };
  const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  out.adapterInfo = {
    vendor: info.vendor, architecture: info.architecture,
    device: info.device, description: info.description,
    isFallbackAdapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter,
  };
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  stage("strip draw");

  const W = 64, H = 64;
  const code = /* wgsl */ `
    struct V { @location(0) pos: vec2f, @location(1) id: u32 };
    struct O { @builtin(position) p: vec4f, @location(0) @interpolate(flat) id: u32 };
    @vertex fn vs(v: V) -> O { var o: O; o.p = vec4f(v.pos, 0.0, 1.0); o.id = v.id; return o; }
    @fragment fn fs(@location(0) @interpolate(flat) id: u32) -> @location(0) vec4u {
      return vec4u(id, 0u, 0u, 1u);
    }`;
  const module = device.createShaderModule({ code });

  // Strip: zig-zag, each triangle a disjoint region (shares only edges).
  const strip = [
    [-0.9, -0.8, 10], [-0.6, 0.8, 11], [-0.2, -0.8, 12], [0.2, 0.8, 13], [0.6, -0.8, 14],
  ];
  // List control: two separate triangles, ids 20..22 and 30..32 in each order.
  const list = [
    [-0.9, -0.9, 20], [-0.1, -0.9, 21], [-0.5, -0.1, 22],
    [0.1, 0.1, 30], [0.9, 0.1, 31], [0.5, 0.9, 32],
  ];

  function pack(verts) {
    const buf = new ArrayBuffer(verts.length * 12);
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    verts.forEach(([x, y, id], i) => { f[i * 3] = x; f[i * 3 + 1] = y; u[i * 3 + 2] = id; });
    const b = device.createBuffer({ size: buf.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, buf);
    return b;
  }

  async function draw(topology, verts) {
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module, entryPoint: "vs",
        buffers: [{ arrayStride: 12, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "uint32" },
        ] }],
      },
      fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8uint" }] },
      primitive: { topology, cullMode: "none" },
    });
    const tex = device.createTexture({ size: [W, H], format: "rgba8uint", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 256 * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, pack(verts));
    pass.draw(verts.length);
    pass.end();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: 256 }, [W, H]);
    device.queue.submit([enc.finish()]);
    stage("mapAsync " + topology);
    await rb.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return (x, y) => {
      const ix = Math.floor((x + 1) / 2 * W), iy = Math.floor((1 - y) / 2 * H);
      return { ndc: [+x.toFixed(3), +y.toFixed(3)], pixel: [ix, iy], value: px[iy * 256 + ix * 4] };
    };
  }

  const centroid = (vs) => [vs.reduce((s, v) => s + v[0], 0) / 3, vs.reduce((s, v) => s + v[1], 0) / 3];

  const readStrip = await draw("triangle-strip", strip);
  out.strip = [0, 1, 2].map((k) => {
    const [x, y] = centroid(strip.slice(k, k + 3));
    const r = readStrip(x, y);
    const hyp = { "v[k]": strip[k][2], "v[k+1]": strip[k + 1][2], "v[k+2]": strip[k + 2][2] };
    r.triangle = k;
    r.matches = Object.keys(hyp).filter((h) => hyp[h] === r.value);
    return r;
  });

  stage("list draw");
  const readList = await draw("triangle-list", list);
  out.list = [0, 1].map((t) => {
    const tri = list.slice(t * 3, t * 3 + 3);
    const [x, y] = centroid(tri);
    const r = readList(x, y);
    r.triangle = t;
    r.firstVertexId = tri[0][2];
    r.isFirst = r.value === tri[0][2];
    return r;
  });

  stage("popErrorScope");
  const err = await device.popErrorScope();
  stage("done");
  if (err) out.errors.push(err.message);
  const verdicts = out.strip.map((s) => s.matches.join("|") || "none");
  out.conclusion = {
    perTriangle: verdicts,
    listFlatIsFirst: out.list.every((l) => l.isFirst),
  };
  return out;
}
