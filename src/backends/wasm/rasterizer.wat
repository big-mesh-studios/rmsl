;; Source of truth for the rasterizer module's bytes -- see rasterizer.md
;; for the algorithm and RASTERIZE_PARAMS in rasterizer.ts for the exact
;; argument list/order this module's "rasterize" export expects.
;;
;; rasterizer.ts imports this file directly; vite.config.ts/vitest.config.ts
;; wire up `compileWat` (src/vite/vite.ts) to compile it to bytes at build
;; time via wabt, so nothing here is shipped or evaluated at runtime.
(module
  (import "vertex" "main" (func $vertexMain))
  (import "fragment" "main" (func $fragmentMain))
  (import "env" "memory" (memory 1))

  ;; ---- generic byte copy: memory[dest..dest+len) = memory[src..src+len) ----
  (func $byteCopy (param $dest i32) (param $src i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (local.get $dest) (local.get $i))
          (i32.load8_u (i32.add (local.get $src) (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue))))

  ;; ---- descriptor/attribute addressing ----
  (func $descAddr (param $base i32) (param $index i32) (param $entryBytes i32) (result i32)
    (i32.add (local.get $base) (i32.mul (local.get $index) (local.get $entryBytes))))

  (func $posComponentAt (param $base i32) (param $vertexIndex i32) (param $comp i32) (result f64)
    (f64.load
      (i32.add
        (i32.add (local.get $base) (i32.mul (local.get $vertexIndex) (i32.const 32)))
        (i32.mul (local.get $comp) (i32.const 8)))))

  (func $varyingComponentAt
    (param $base i32) (param $varyingBytes i32) (param $vertexIndex i32)
    (param $recordOffset i32) (param $comp i32) (result f64)
    (f64.load
      (i32.add
        (i32.add
          (i32.add (local.get $base) (i32.mul (local.get $vertexIndex) (local.get $varyingBytes)))
          (local.get $recordOffset))
        (i32.mul (local.get $comp) (i32.const 8)))))

  ;; ---- clip scratch (18 vertex slots: two halves of 9 for the polygon being clipped) ----
  (func $scratchVertexAddr (param $clipScratchBase i32) (param $varyingBytes i32) (param $slot i32) (result i32)
    (i32.add
      (local.get $clipScratchBase)
      (i32.mul (local.get $slot) (i32.add (i32.const 32) (local.get $varyingBytes)))))

  (func $scratchVaryingAddr (param $clipScratchBase i32) (param $varyingBytes i32) (param $slot i32) (result i32)
    (i32.add
      (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $slot))
      (i32.const 32)))

  ;; ---- vertex pass ----
  (func $copyAttributesIn
    (param $attrSrcBase i32) (param $attrStrideBytes i32)
    (param $attrDescBase i32) (param $attrDescCount i32) (param $vertexIndex i32)
    (local $d i32) (local $entry i32)
    (local.set $d (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $d) (local.get $attrDescCount)))
        (local.set $entry (call $descAddr (local.get $attrDescBase) (local.get $d) (i32.const 12)))
        (call $byteCopy
          (i32.load (i32.add (local.get $entry) (i32.const 4)))    ;; destAddress
          (i32.add
            (i32.add (local.get $attrSrcBase) (i32.mul (local.get $vertexIndex) (local.get $attrStrideBytes)))
            (i32.load (local.get $entry)))                          ;; + srcOffset
          (i32.load (i32.add (local.get $entry) (i32.const 8))))    ;; sizeBytes
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  (func $copyVaryingsOut
    (param $varyingsOutBase i32) (param $varyingBytes i32)
    (param $varyingDescBase i32) (param $varyingDescCount i32) (param $vertexIndex i32)
    (local $d i32) (local $entry i32)
    (local.set $d (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $d) (local.get $varyingDescCount)))
        (local.set $entry (call $descAddr (local.get $varyingDescBase) (local.get $d) (i32.const 20)))
        (call $byteCopy
          (i32.add
            (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $vertexIndex) (local.get $varyingBytes)))
            (i32.load (local.get $entry)))                          ;; + recordOffset
          (i32.load (i32.add (local.get $entry) (i32.const 4)))    ;; vertexSrcAddress
          (i32.load (i32.add (local.get $entry) (i32.const 12))))  ;; sizeBytes
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  ;; ---- clip pass (Sutherland-Hodgman against WebGPU's clip volume, plane by plane) ----
  ;; Copies original vertex $vertexIndex, position and varyings, into scratch slot $slot.
  (func $vertexToScratch
    (param $positionsOutBase i32) (param $varyingsOutBase i32) (param $varyingBytes i32)
    (param $clipScratchBase i32) (param $vertexIndex i32) (param $slot i32)
    (call $byteCopy
      (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $slot))
      (i32.add (local.get $positionsOutBase) (i32.mul (local.get $vertexIndex) (i32.const 32)))
      (i32.const 32))
    (call $byteCopy
      (call $scratchVaryingAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $slot))
      (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $vertexIndex) (local.get $varyingBytes)))
      (local.get $varyingBytes)))

  ;; Copies scratch slot $from, position and varyings, into scratch slot $to.
  (func $scratchToScratch (param $clipScratchBase i32) (param $varyingBytes i32) (param $from i32) (param $to i32)
    (call $byteCopy
      (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $to))
      (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $from))
      (i32.add (i32.const 32) (local.get $varyingBytes))))

  ;; Writes the vertex a fraction $t of the way from scratch slot $a to slot $b
  ;; into slot $to. A flat varying is copied from $firstVaryings, the varyings
  ;; of the first vertex of the triangle being clipped, as no value lies between two.
  ;; A position whose four components all lie within 2^-48 of the largest sum the cut
  ;; takes, a component of $a and its step added by size, is rounding alone and is
  ;; vec4(0). One whose w alone is 0 keeps its x, y and z for the next plane to cut.
  (func $lerpToScratch
    (param $clipScratchBase i32) (param $varyingBytes i32) (param $varyingDescBase i32) (param $varyingDescCount i32)
    (param $a i32) (param $b i32) (param $t f64) (param $firstVaryings i32) (param $to i32)
    (local $c i32) (local $aAddr i32) (local $bAddr i32) (local $toAddr i32)
    (local $d i32) (local $entry i32) (local $recordOffset i32) (local $numComponents i32) (local $offset i32)
    (local $step f64) (local $rounding f64) (local $size f64)
    (local.set $aAddr (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $a)))
    (local.set $bAddr (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $b)))
    (local.set $toAddr (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $to)))
    (local.set $c (i32.const 0))
    (block $posBreak
      (loop $posContinue
        (br_if $posBreak (i32.ge_s (local.get $c) (i32.const 4)))
        (local.set $offset (i32.mul (local.get $c) (i32.const 8)))
        (local.set $step
          (f64.mul
            (f64.sub (f64.load (i32.add (local.get $bAddr) (local.get $offset))) (f64.load (i32.add (local.get $aAddr) (local.get $offset))))
            (local.get $t)))
        (f64.store (i32.add (local.get $toAddr) (local.get $offset))
          (f64.add (f64.load (i32.add (local.get $aAddr) (local.get $offset))) (local.get $step)))
        (local.set $rounding
          (f64.max (local.get $rounding)
            (f64.add (f64.abs (f64.load (i32.add (local.get $aAddr) (local.get $offset)))) (f64.abs (local.get $step)))))
        (local.set $size
          (f64.max (local.get $size) (f64.abs (f64.load (i32.add (local.get $toAddr) (local.get $offset))))))
        (local.set $c (i32.add (local.get $c) (i32.const 1)))
        (br $posContinue)))
    (if (f64.le (local.get $size) (f64.mul (local.get $rounding) (f64.const 0x1p-48)))
      (then
        (f64.store (local.get $toAddr) (f64.const 0))
        (f64.store (i32.add (local.get $toAddr) (i32.const 8)) (f64.const 0))
        (f64.store (i32.add (local.get $toAddr) (i32.const 16)) (f64.const 0))
        (f64.store (i32.add (local.get $toAddr) (i32.const 24)) (f64.const 0))))
    (local.set $d (i32.const 0))
    (block $varyingBreak
      (loop $varyingContinue
        (br_if $varyingBreak (i32.ge_s (local.get $d) (local.get $varyingDescCount)))
        (local.set $entry (call $descAddr (local.get $varyingDescBase) (local.get $d) (i32.const 20)))
        (local.set $recordOffset (i32.load (local.get $entry)))
        (local.set $numComponents (i32.div_s (i32.load (i32.add (local.get $entry) (i32.const 12))) (i32.const 8)))
        (if (i32.load (i32.add (local.get $entry) (i32.const 16)))
          (then
            (call $byteCopy
              (i32.add (i32.add (local.get $toAddr) (i32.const 32)) (local.get $recordOffset))
              (i32.add (local.get $firstVaryings) (local.get $recordOffset))
              (i32.load (i32.add (local.get $entry) (i32.const 12))))
            (local.set $numComponents (i32.const 0))))
        (local.set $c (i32.const 0))
        (block $componentBreak
          (loop $componentContinue
            (br_if $componentBreak (i32.ge_s (local.get $c) (local.get $numComponents)))
            (local.set $offset (i32.add (i32.add (i32.const 32) (local.get $recordOffset)) (i32.mul (local.get $c) (i32.const 8))))
            (f64.store (i32.add (local.get $toAddr) (local.get $offset))
              (f64.add (f64.load (i32.add (local.get $aAddr) (local.get $offset)))
                (f64.mul
                  (f64.sub (f64.load (i32.add (local.get $bAddr) (local.get $offset))) (f64.load (i32.add (local.get $aAddr) (local.get $offset))))
                  (local.get $t))))
            (local.set $c (i32.add (local.get $c) (i32.const 1)))
            (br $componentContinue)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $varyingContinue))))

  ;; How far the position at $addr lies inside plane $plane of the clip volume,
  ;; negative outside it: w + x, w - x, w + y, w - y, z and w - z, in that order.
  (func $planeDistance (param $addr i32) (param $plane i32) (result f64)
    (local $w f64) (local $c f64)
    (local.set $w (f64.load (i32.add (local.get $addr) (i32.const 24))))
    (if (i32.eq (local.get $plane) (i32.const 4))
      (then (return (f64.load (i32.add (local.get $addr) (i32.const 16))))))
    ;; Planes 0 and 1 read x, 2 and 3 read y, 5 reads z; an odd plane subtracts it.
    (local.set $c
      (f64.load (i32.add (local.get $addr)
        (select (i32.const 16) (i32.mul (i32.shr_u (local.get $plane) (i32.const 1)) (i32.const 8))
          (i32.eq (local.get $plane) (i32.const 5))))))
    (select
      (f64.sub (local.get $w) (local.get $c))
      (f64.add (local.get $w) (local.get $c))
      (i32.and (local.get $plane) (i32.const 1))))

  ;; Clips the polygon of $count vertices in scratch slots from $src against
  ;; plane $plane into the slots from $dst, and returns how many it wrote.
  (func $clipPolygon
    (param $clipScratchBase i32) (param $varyingBytes i32) (param $varyingDescBase i32) (param $varyingDescCount i32)
    (param $src i32) (param $count i32) (param $dst i32) (param $plane i32) (param $firstVaryings i32) (result i32)
    (local $i i32) (local $a i32) (local $b i32) (local $da f64) (local $db f64) (local $out i32)
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $i) (local.get $count)))
        (local.set $a (i32.add (local.get $src) (local.get $i)))
        (local.set $b (i32.add (local.get $src) (i32.rem_s (i32.add (local.get $i) (i32.const 1)) (local.get $count))))
        (local.set $da (call $planeDistance (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $a)) (local.get $plane)))
        (local.set $db (call $planeDistance (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $b)) (local.get $plane)))
        ;; Rounding can make a cut polygon slightly concave; it keeps 9 vertices at most, the slots of a half.
        (if (i32.and (f64.ge (local.get $da) (f64.const 0)) (i32.lt_s (local.get $out) (i32.const 9)))
          (then
            (call $scratchToScratch (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $a) (i32.add (local.get $dst) (local.get $out)))
            (local.set $out (i32.add (local.get $out) (i32.const 1)))))
        (if (i32.and (i32.ne (f64.ge (local.get $da) (f64.const 0)) (f64.ge (local.get $db) (f64.const 0))) (i32.lt_s (local.get $out) (i32.const 9)))
          (then
            ;; A cut runs from the vertex inside, so a w far smaller than the other's keeps its digits.
            (if (f64.ge (local.get $da) (f64.const 0))
              (then
                (call $lerpToScratch (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $varyingDescBase) (local.get $varyingDescCount)
                  (local.get $a) (local.get $b)
                  (f64.div (local.get $da) (f64.sub (local.get $da) (local.get $db)))
                  (local.get $firstVaryings) (i32.add (local.get $dst) (local.get $out))))
              (else
                (call $lerpToScratch (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $varyingDescBase) (local.get $varyingDescCount)
                  (local.get $b) (local.get $a)
                  (f64.div (local.get $db) (f64.sub (local.get $db) (local.get $da)))
                  (local.get $firstVaryings) (i32.add (local.get $dst) (local.get $out)))))
            (local.set $out (i32.add (local.get $out) (i32.const 1)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $out))

  (func $emitTriangleFromScratch
    (param $clipScratchBase i32) (param $varyingBytes i32)
    (param $clippedPositionsOutBase i32) (param $clippedVaryingsOutBase i32)
    (param $clippedVertexCount i32) (param $slot0 i32) (param $slot1 i32) (param $slot2 i32) (result i32)
    (local $i i32) (local $slot i32)
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $i) (i32.const 3)))
        (local.set $slot
          (if (result i32) (i32.eqz (local.get $i))
            (then (local.get $slot0))
            (else (if (result i32) (i32.eq (local.get $i) (i32.const 1)) (then (local.get $slot1)) (else (local.get $slot2))))))
        (call $byteCopy
          (i32.add (local.get $clippedPositionsOutBase)
            (i32.mul (i32.add (local.get $clippedVertexCount) (local.get $i)) (i32.const 32)))
          (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $slot))
          (i32.const 32))
        (call $byteCopy
          (i32.add (local.get $clippedVaryingsOutBase)
            (i32.mul (i32.add (local.get $clippedVertexCount) (local.get $i)) (local.get $varyingBytes)))
          (call $scratchVaryingAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $slot))
          (local.get $varyingBytes))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (i32.add (local.get $clippedVertexCount) (i32.const 3)))

  ;; ---- triangle pass ----
  ;; Whether an edge running (dx, dy), in a triangle wound so its inside is where
  ;; every edge function is positive, owns the pixel centres on it: a left edge,
  ;; running up the screen, or a top edge, running right, by WebGPU's top-left rule.
  ;; Two triangles run a shared edge in opposite directions, so exactly one owns it.
  (func $ownsEdge (param $dx f64) (param $dy f64) (result i32)
    (i32.or
      (f64.lt (local.get $dy) (f64.const 0))
      (i32.and (f64.eq (local.get $dy) (f64.const 0)) (f64.gt (local.get $dx) (f64.const 0)))))

  ;; Whether a pixel centre lies on the inside of an edge whose function, wound
  ;; positive inside, is $f, an edge the triangle $owns included.
  (func $insideEdge (param $f f64) (param $owns i32) (result i32)
    (i32.or
      (f64.gt (local.get $f) (f64.const 0))
      (i32.and (f64.eq (local.get $f) (f64.const 0)) (local.get $owns))))

  ;; A clip coordinate over w; 0 for a vertex at vec4(0), as WebGPU places it.
  (func $ndc (param $c f64) (param $w f64) (result f64)
    (select (f64.const 0) (f64.div (local.get $c) (local.get $w)) (f64.eq (local.get $w) (f64.const 0))))

  ;; The perspective weight of a vertex at $w for each unit of its barycentric, times the
  ;; least w, $wNear. When the inverse of $wNear is not $finite, it is 1 for the vertex
  ;; of that w, which $isNear says, and 0 for the others.
  (func $nearWeight (param $finite i32) (param $wNear f64) (param $w f64) (param $isNear i32) (result f64)
    (if (result f64) (local.get $finite)
      (then (f64.div (local.get $wNear) (local.get $w)))
      (else (f64.convert_i32_u (local.get $isNear)))))

  (func $screenX (param $clippedPositionsOutBase i32) (param $widthF f64) (param $vertexIndex i32) (result f64)
    (f64.mul
      (f64.add
        (f64.mul
          (call $ndc
            (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $vertexIndex) (i32.const 0))
            (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $vertexIndex) (i32.const 3)))
          (f64.const 0.5))
        (f64.const 0.5))
      (local.get $widthF)))

  (func $screenY (param $clippedPositionsOutBase i32) (param $heightF f64) (param $vertexIndex i32) (result f64)
    (f64.mul
      (f64.sub
        (f64.const 1)
        (f64.add
          (f64.mul
            (call $ndc
              (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $vertexIndex) (i32.const 1))
              (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $vertexIndex) (i32.const 3)))
            (f64.const 0.5))
          (f64.const 0.5)))
      (local.get $heightF)))

  (func $edgeFn (param $ax f64) (param $ay f64) (param $bx f64) (param $by f64) (param $px f64) (param $py f64) (result f64)
    (f64.sub
      (f64.mul (f64.sub (local.get $ax) (local.get $px)) (f64.sub (local.get $by) (local.get $py)))
      (f64.mul (f64.sub (local.get $ay) (local.get $py)) (f64.sub (local.get $bx) (local.get $px)))))

  (func $interpolateVaryings
    (param $varyingDescBase i32) (param $varyingDescCount i32)
    (param $clippedVaryingsOutBase i32) (param $varyingBytes i32)
    (param $t0 i32) (param $t1 i32) (param $t2 i32)
    (param $p0 f64) (param $p1 f64) (param $p2 f64) (param $weights f64)
    (local $d i32) (local $entry i32) (local $recordOffset i32) (local $fragmentDestAddress i32)
    (local $numComponents i32) (local $c i32)
    (local.set $d (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $d) (local.get $varyingDescCount)))
        (local.set $entry (call $descAddr (local.get $varyingDescBase) (local.get $d) (i32.const 20)))
        (local.set $recordOffset (i32.load (local.get $entry)))
        (local.set $fragmentDestAddress (i32.load (i32.add (local.get $entry) (i32.const 8))))
        (local.set $numComponents (i32.div_s (i32.load (i32.add (local.get $entry) (i32.const 12))) (i32.const 8)))
        ;; A flat varying is copied as the triangle's vertices all hold it, and none is interpolated.
        (if (i32.load (i32.add (local.get $entry) (i32.const 16)))
          (then
            (call $byteCopy
              (local.get $fragmentDestAddress)
              (i32.add
                (i32.add (local.get $clippedVaryingsOutBase) (i32.mul (local.get $t0) (local.get $varyingBytes)))
                (local.get $recordOffset))
              (i32.load (i32.add (local.get $entry) (i32.const 12))))
            (local.set $numComponents (i32.const 0))))
        (local.set $c (i32.const 0))
        (block $innerBreak
          (loop $innerContinue
            (br_if $innerBreak (i32.ge_s (local.get $c) (local.get $numComponents)))
            (f64.store
              (i32.add (local.get $fragmentDestAddress) (i32.mul (local.get $c) (i32.const 8)))
              (f64.div
                (f64.add
                  (f64.add
                    (f64.mul (local.get $p0)
                      (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t0) (local.get $recordOffset) (local.get $c)))
                    (f64.mul (local.get $p1)
                      (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t1) (local.get $recordOffset) (local.get $c))))
                  (f64.mul (local.get $p2)
                    (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t2) (local.get $recordOffset) (local.get $c))))
                (local.get $weights)))
            (local.set $c (i32.add (local.get $c) (i32.const 1)))
            (br $innerContinue)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  ;; ---- rasterize: vertex pass, then the clip and triangle passes, one triangle at a time ----
  (func $rasterize
    (param $vertexCount i32) (param $attrSrcBase i32) (param $attrStrideBytes i32)
    (param $attrDescBase i32) (param $attrDescCount i32) (param $vertexPositionAddress i32)
    (param $positionsOutBase i32) (param $width i32) (param $height i32)
    (param $fragmentValueAddress i32) (param $outputBase i32) (param $varyingBytes i32)
    (param $varyingDescBase i32) (param $varyingDescCount i32) (param $varyingsOutBase i32)
    (param $clipScratchBase i32) (param $clippedPositionsOutBase i32) (param $clippedVaryingsOutBase i32)
    (param $depthBufferBase i32)
    ;; What the fragment stage writes or reads besides its colour: each address
    ;; is used only when the flag after it is 1.
    (param $writesColour i32)
    (param $fragCoordAddress i32) (param $readsFragCoord i32)
    (param $fragDepthAddress i32) (param $usesDepth i32) (param $writesDepth i32)
    (param $discardAddress i32) (param $mayDiscard i32)

    (local $i i32) (local $t i32) (local $t1 i32) (local $t2 i32)
    (local $widthF f64) (local $heightF f64)
    (local $outCount i32) (local $clippedVertexCount i32)
    (local $s0x f64) (local $s0y f64) (local $s1x f64) (local $s1y f64) (local $s2x f64) (local $s2y f64)
    (local $w0 f64) (local $w1 f64) (local $w2 f64)
    (local $r0 f64) (local $r1 f64) (local $r2 f64) (local $near i32) (local $wNear f64) (local $finite i32)
    (local $depth0 f64) (local $depth1 f64) (local $depth2 f64)
    (local $area f64)
    (local $minX i32) (local $maxX i32) (local $minY i32) (local $maxY i32)
    (local $x i32) (local $y i32) (local $px f64) (local $py f64) (local $pixelIndex i32)
    (local $e0 f64) (local $e1 f64) (local $e2 f64)
    (local $b0 f64) (local $b1 f64) (local $b2 f64) (local $p0 f64) (local $p1 f64) (local $p2 f64) (local $weights f64)
    (local $pixelDepth f64) (local $depthAddr i32) (local $fragDepth f64)
    (local $wind f64) (local $owns0 i32) (local $owns1 i32) (local $owns2 i32)
    (local $nearest f64) (local $farthest f64) (local $crossesDepthRange i32)
    (local $src i32) (local $plane i32) (local $tri i32) (local $tri1 i32) (local $tri2 i32)

    (local.set $widthF (f64.convert_i32_s (local.get $width)))
    (local.set $heightF (f64.convert_i32_s (local.get $height)))

    ;; ---- vertex pass ----
    (local.set $i (i32.const 0))
    (block $vertexBreak
      (loop $vertexContinue
        (br_if $vertexBreak (i32.ge_s (local.get $i) (local.get $vertexCount)))
        (call $copyAttributesIn (local.get $attrSrcBase) (local.get $attrStrideBytes) (local.get $attrDescBase) (local.get $attrDescCount) (local.get $i))
        (call $vertexMain)
        (call $byteCopy
          (i32.add (local.get $positionsOutBase) (i32.mul (local.get $i) (i32.const 32)))
          (local.get $vertexPositionAddress)
          (i32.const 32))
        (call $copyVaryingsOut (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $varyingDescBase) (local.get $varyingDescCount) (local.get $i))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $vertexContinue)))

    ;; ---- clip and triangle pass, one triangle at a time ----
    ;; A triangle wholly in front of the eye passes as it is. One with a vertex near
    ;; or behind it is clipped to the clip volume, two halves of the scratch, nine
    ;; slots each, taking turns, and fanned out of the polygon left; the fan is
    ;; rasterized before the next triangle is clipped.
    (local.set $tri (i32.const 0))
    (block $clipBreak
      (loop $clipContinue
        (local.set $tri1 (i32.add (local.get $tri) (i32.const 1)))
        (local.set $tri2 (i32.add (local.get $tri) (i32.const 2)))
        (br_if $clipBreak (i32.ge_s (local.get $tri2) (local.get $vertexCount)))
        (call $vertexToScratch (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $clipScratchBase) (local.get $tri) (i32.const 0))
        (call $vertexToScratch (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $clipScratchBase) (local.get $tri1) (i32.const 1))
        (call $vertexToScratch (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $clipScratchBase) (local.get $tri2) (i32.const 2))
        (local.set $clippedVertexCount (i32.const 0))
        (local.set $outCount (i32.const 3))
        (local.set $src (i32.const 0))
        (if (i32.eqz
              (i32.and
                (i32.and
                  (f64.gt (call $posComponentAt (local.get $positionsOutBase) (local.get $tri) (i32.const 3)) (f64.const 1e-5))
                  (f64.gt (call $posComponentAt (local.get $positionsOutBase) (local.get $tri1) (i32.const 3)) (f64.const 1e-5)))
                (f64.gt (call $posComponentAt (local.get $positionsOutBase) (local.get $tri2) (i32.const 3)) (f64.const 1e-5))))
          (then
            (local.set $plane (i32.const 0))
            (block $planeBreak
              (loop $planeContinue
                (br_if $planeBreak (i32.or (i32.ge_s (local.get $plane) (i32.const 6)) (i32.lt_s (local.get $outCount) (i32.const 3))))
                (local.set $outCount
                  (call $clipPolygon (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $varyingDescBase) (local.get $varyingDescCount)
                    (local.get $src) (local.get $outCount) (i32.sub (i32.const 9) (local.get $src)) (local.get $plane)
                    (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $tri) (local.get $varyingBytes)))))
                (local.set $src (i32.sub (i32.const 9) (local.get $src)))
                (local.set $plane (i32.add (local.get $plane) (i32.const 1)))
                (br $planeContinue)))))
        (local.set $i (i32.const 1))
        (block $fanBreak
          (loop $fanContinue
            (br_if $fanBreak (i32.ge_s (i32.add (local.get $i) (i32.const 1)) (local.get $outCount)))
            (local.set $clippedVertexCount
              (call $emitTriangleFromScratch (local.get $clipScratchBase) (local.get $varyingBytes)
                (local.get $clippedPositionsOutBase) (local.get $clippedVaryingsOutBase)
                (local.get $clippedVertexCount) (local.get $src) (i32.add (local.get $src) (local.get $i)) (i32.add (local.get $src) (i32.add (local.get $i) (i32.const 1)))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $fanContinue)))
        ;; ---- triangle pass ----
        (local.set $t (i32.const 0))
        (block $triBreak
          (loop $triContinue
            (local.set $t1 (i32.add (local.get $t) (i32.const 1)))
            (local.set $t2 (i32.add (local.get $t) (i32.const 2)))
            (br_if $triBreak (i32.ge_s (local.get $t2) (local.get $clippedVertexCount)))

            (block $skipDegenerate
              (local.set $s0x (call $screenX (local.get $clippedPositionsOutBase) (local.get $widthF) (local.get $t)))
              (local.set $s0y (call $screenY (local.get $clippedPositionsOutBase) (local.get $heightF) (local.get $t)))
              (local.set $s1x (call $screenX (local.get $clippedPositionsOutBase) (local.get $widthF) (local.get $t1)))
              (local.set $s1y (call $screenY (local.get $clippedPositionsOutBase) (local.get $heightF) (local.get $t1)))
              (local.set $s2x (call $screenX (local.get $clippedPositionsOutBase) (local.get $widthF) (local.get $t2)))
              (local.set $s2y (call $screenY (local.get $clippedPositionsOutBase) (local.get $heightF) (local.get $t2)))
              (local.set $w0 (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t) (i32.const 3)))
              (local.set $w1 (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t1) (i32.const 3)))
              (local.set $w2 (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t2) (i32.const 3)))
              ;; Perspective weighs vertex i by its barycentric over its w, here times the least w, so
              ;; no weight overflows or underflows at any scale. A least w whose inverse is not
              ;; finite, as at vec4(0), takes all the weight.
              (local.set $near (i32.const 0))
              (local.set $wNear (local.get $w0))
              (if (f64.lt (local.get $w1) (local.get $wNear))
                (then (local.set $near (i32.const 1)) (local.set $wNear (local.get $w1))))
              (if (f64.lt (local.get $w2) (local.get $wNear))
                (then (local.set $near (i32.const 2)) (local.set $wNear (local.get $w2))))
              (local.set $finite
                (i32.and
                  (f64.gt (f64.div (f64.const 1) (local.get $wNear)) (f64.const 0))
                  (f64.lt (f64.div (f64.const 1) (local.get $wNear)) (f64.const inf))))
              (local.set $r0 (call $nearWeight (local.get $finite) (local.get $wNear) (local.get $w0) (i32.eq (local.get $near) (i32.const 0))))
              (local.set $r1 (call $nearWeight (local.get $finite) (local.get $wNear) (local.get $w1) (i32.eq (local.get $near) (i32.const 1))))
              (local.set $r2 (call $nearWeight (local.get $finite) (local.get $wNear) (local.get $w2) (i32.eq (local.get $near) (i32.const 2))))
              (local.set $depth0 (call $ndc (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t) (i32.const 2)) (local.get $w0)))
              (local.set $depth1 (call $ndc (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t1) (i32.const 2)) (local.get $w1)))
              (local.set $depth2 (call $ndc (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t2) (i32.const 2)) (local.get $w2)))
              (local.set $area
                (f64.sub
                  (f64.mul (f64.sub (local.get $s1x) (local.get $s0x)) (f64.sub (local.get $s2y) (local.get $s0y)))
                  (f64.mul (f64.sub (local.get $s1y) (local.get $s0y)) (f64.sub (local.get $s2x) (local.get $s0x)))))
              (br_if $skipDegenerate (f64.eq (local.get $area) (f64.const 0)))
              ;; A triangle wholly outside depth 0 to 1 draws nothing. One that crosses it drops each
              ;; pixel outside it, which is what clipping there drops, as depth over w is affine in
              ;; screen space; one inside it draws every pixel, whatever the rounding of its depth.
              (local.set $nearest (f64.min (f64.min (local.get $depth0) (local.get $depth1)) (local.get $depth2)))
              (local.set $farthest (f64.max (f64.max (local.get $depth0) (local.get $depth1)) (local.get $depth2)))
              (br_if $skipDegenerate
                (i32.or (f64.lt (local.get $farthest) (f64.const 0)) (f64.gt (local.get $nearest) (f64.const 1))))
              (local.set $crossesDepthRange
                (i32.or (f64.lt (local.get $nearest) (f64.const 0)) (f64.gt (local.get $farthest) (f64.const 1))))
              ;; Each edge function, times $wind, is positive inside whichever way the triangle winds.
              (local.set $wind (select (f64.const 1) (f64.const -1) (f64.gt (local.get $area) (f64.const 0))))
              (local.set $owns0 (call $ownsEdge
                (f64.mul (f64.sub (local.get $s2x) (local.get $s1x)) (local.get $wind))
                (f64.mul (f64.sub (local.get $s2y) (local.get $s1y)) (local.get $wind))))
              (local.set $owns1 (call $ownsEdge
                (f64.mul (f64.sub (local.get $s0x) (local.get $s2x)) (local.get $wind))
                (f64.mul (f64.sub (local.get $s0y) (local.get $s2y)) (local.get $wind))))
              (local.set $owns2 (call $ownsEdge
                (f64.mul (f64.sub (local.get $s1x) (local.get $s0x)) (local.get $wind))
                (f64.mul (f64.sub (local.get $s1y) (local.get $s0y)) (local.get $wind))))

              (local.set $minX (i32.trunc_sat_f64_s (f64.max (f64.floor (f64.min (f64.min (local.get $s0x) (local.get $s1x)) (local.get $s2x))) (f64.const 0))))
              (local.set $maxX (i32.trunc_sat_f64_s (f64.min (f64.ceil (f64.max (f64.max (local.get $s0x) (local.get $s1x)) (local.get $s2x))) (f64.sub (local.get $widthF) (f64.const 1)))))
              (local.set $minY (i32.trunc_sat_f64_s (f64.max (f64.floor (f64.min (f64.min (local.get $s0y) (local.get $s1y)) (local.get $s2y))) (f64.const 0))))
              (local.set $maxY (i32.trunc_sat_f64_s (f64.min (f64.ceil (f64.max (f64.max (local.get $s0y) (local.get $s1y)) (local.get $s2y))) (f64.sub (local.get $heightF) (f64.const 1)))))

              (local.set $y (local.get $minY))
              (block $yBreak
                (loop $yContinue
                  (br_if $yBreak (i32.gt_s (local.get $y) (local.get $maxY)))
                  (local.set $x (local.get $minX))
                  (block $xBreak
                    (loop $xContinue
                      (br_if $xBreak (i32.gt_s (local.get $x) (local.get $maxX)))
                      (local.set $px (f64.add (f64.convert_i32_s (local.get $x)) (f64.const 0.5)))
                      (local.set $py (f64.add (f64.convert_i32_s (local.get $y)) (f64.const 0.5)))
                      (local.set $pixelIndex (i32.add (i32.mul (local.get $y) (local.get $width)) (local.get $x)))
                      (local.set $e0 (call $edgeFn (local.get $s1x) (local.get $s1y) (local.get $s2x) (local.get $s2y) (local.get $px) (local.get $py)))
                      (local.set $e1 (call $edgeFn (local.get $s2x) (local.get $s2y) (local.get $s0x) (local.get $s0y) (local.get $px) (local.get $py)))
                      (local.set $e2 (call $edgeFn (local.get $s0x) (local.get $s0y) (local.get $s1x) (local.get $s1y) (local.get $px) (local.get $py)))
                      ;; covered when the pixel centre is inside all three edges
                      (if
                        (i32.and
                          (i32.and
                            (call $insideEdge (f64.mul (local.get $e0) (local.get $wind)) (local.get $owns0))
                            (call $insideEdge (f64.mul (local.get $e1) (local.get $wind)) (local.get $owns1)))
                          (call $insideEdge (f64.mul (local.get $e2) (local.get $wind)) (local.get $owns2)))
                        (then
                          (local.set $b0 (f64.div (local.get $e0) (local.get $area)))
                          (local.set $b1 (f64.div (local.get $e1) (local.get $area)))
                          (local.set $b2 (f64.div (local.get $e2) (local.get $area)))
                          (local.set $p0 (f64.mul (local.get $b0) (local.get $r0)))
                          (local.set $p1 (f64.mul (local.get $b1) (local.get $r1)))
                          (local.set $p2 (f64.mul (local.get $b2) (local.get $r2)))
                          (local.set $weights (f64.add (f64.add (local.get $p0) (local.get $p1)) (local.get $p2)))
                          ;; On the edge opposite the vertex that takes all the weight, that vertex's barycentric is 0 too.
                          (if (f64.eq (local.get $weights) (f64.const 0))
                            (then
                              (local.set $p0 (f64.convert_i32_u (i32.eq (local.get $near) (i32.const 0))))
                              (local.set $p1 (f64.convert_i32_u (i32.eq (local.get $near) (i32.const 1))))
                              (local.set $p2 (f64.convert_i32_u (i32.eq (local.get $near) (i32.const 2))))
                              (local.set $weights (f64.const 1))))
                          (local.set $pixelDepth
                            (f64.add
                              (f64.add (f64.mul (local.get $b0) (local.get $depth0)) (f64.mul (local.get $b1) (local.get $depth1)))
                              (f64.mul (local.get $b2) (local.get $depth2))))
                          (local.set $depthAddr (i32.add (local.get $depthBufferBase) (i32.mul (local.get $pixelIndex) (i32.const 8))))
                          (block $skipPixel
                            (br_if $skipPixel
                              (i32.and (local.get $crossesDepthRange)
                                (i32.or (f64.lt (local.get $pixelDepth) (f64.const 0)) (f64.gt (local.get $pixelDepth) (f64.const 1)))))
                            ;; LEQUAL depth test; a stage that writes its depth is tested once it has run.
                            (br_if $skipPixel
                              (i32.and (i32.eqz (local.get $writesDepth))
                                (f64.gt (local.get $pixelDepth) (f64.load (local.get $depthAddr)))))
                            (call $interpolateVaryings
                              (local.get $varyingDescBase) (local.get $varyingDescCount)
                              (local.get $clippedVaryingsOutBase) (local.get $varyingBytes)
                              (local.get $t) (local.get $t1) (local.get $t2)
                              (local.get $p0) (local.get $p1) (local.get $p2) (local.get $weights))
                            (if (local.get $readsFragCoord)
                              (then
                                (f64.store (local.get $fragCoordAddress) (local.get $px))
                                (f64.store (i32.add (local.get $fragCoordAddress) (i32.const 8)) (local.get $py))))
                            (if (local.get $mayDiscard) (then (i32.store (local.get $discardAddress) (i32.const 0))))
                            ;; A stage that leaves its depth unwritten keeps the interpolated one.
                            (if (local.get $usesDepth) (then (f64.store (local.get $fragDepthAddress) (local.get $pixelDepth))))
                            (call $fragmentMain)
                            ;; A fragment that discards leaves the pixel and its depth as they were.
                            (if (local.get $mayDiscard) (then (br_if $skipPixel (i32.load (local.get $discardAddress)))))
                            ;; A written depth is clamped to the depth range, as WebGPU clamps it.
                            (local.set $fragDepth
                              (select
                                (f64.min (f64.max (f64.load (local.get $fragDepthAddress)) (f64.const 0)) (f64.const 1))
                                (local.get $pixelDepth)
                                (local.get $writesDepth)))
                            (br_if $skipPixel
                              (i32.and (local.get $writesDepth)
                                (f64.gt (local.get $fragDepth) (f64.load (local.get $depthAddr)))))
                            (f64.store (local.get $depthAddr) (local.get $fragDepth))
                            ;; A fragment stage that writes no colour leaves the pixel as it is.
                            (if (local.get $writesColour)
                              (then
                                (call $byteCopy
                                  (i32.add (local.get $outputBase) (i32.mul (local.get $pixelIndex) (i32.const 32)))
                                  (local.get $fragmentValueAddress)
                                  (i32.const 32)))))))
                      (local.set $x (i32.add (local.get $x) (i32.const 1)))
                      (br $xContinue)))
                  (local.set $y (i32.add (local.get $y) (i32.const 1)))
                  (br $yContinue))))

            (local.set $t (i32.add (local.get $t) (i32.const 3)))
            (br $triContinue)))
        (local.set $tri (i32.add (local.get $tri) (i32.const 3)))
        (br $clipContinue))))

  (export "rasterize" (func $rasterize)))
