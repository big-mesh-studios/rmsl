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

  ;; ---- clip scratch (up to 4 vertices: the intermediate clipped polygon) ----
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
        (local.set $entry (call $descAddr (local.get $varyingDescBase) (local.get $d) (i32.const 16)))
        (call $byteCopy
          (i32.add
            (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $vertexIndex) (local.get $varyingBytes)))
            (i32.load (local.get $entry)))                          ;; + recordOffset
          (i32.load (i32.add (local.get $entry) (i32.const 4)))    ;; vertexSrcAddress
          (i32.load (i32.add (local.get $entry) (i32.const 12))))  ;; sizeBytes
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  ;; ---- clip pass (single-plane Sutherland-Hodgman against w > W_CLIP_EPS) ----
  (func $vertexToScratch
    (param $positionsOutBase i32) (param $varyingsOutBase i32) (param $varyingBytes i32)
    (param $clipScratchBase i32) (param $vertexIndex i32) (param $outCount i32) (result i32)
    (call $byteCopy
      (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $outCount))
      (i32.add (local.get $positionsOutBase) (i32.mul (local.get $vertexIndex) (i32.const 32)))
      (i32.const 32))
    (call $byteCopy
      (call $scratchVaryingAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $outCount))
      (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $vertexIndex) (local.get $varyingBytes)))
      (local.get $varyingBytes))
    (i32.add (local.get $outCount) (i32.const 1)))

  (func $interpVertexToScratch
    (param $positionsOutBase i32) (param $varyingsOutBase i32) (param $varyingBytes i32)
    (param $clipScratchBase i32) (param $wholeRecordComponents i32)
    (param $a i32) (param $b i32) (param $t f64) (param $outCount i32) (result i32)
    (local $c i32) (local $vertexAddr i32) (local $varyingAddr i32)
    (local.set $vertexAddr (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $outCount)))
    (local.set $c (i32.const 0))
    (block $posBreak
      (loop $posContinue
        (br_if $posBreak (i32.ge_s (local.get $c) (i32.const 4)))
        (f64.store
          (i32.add (local.get $vertexAddr) (i32.mul (local.get $c) (i32.const 8)))
          (f64.add
            (call $posComponentAt (local.get $positionsOutBase) (local.get $a) (local.get $c))
            (f64.mul
              (f64.sub
                (call $posComponentAt (local.get $positionsOutBase) (local.get $b) (local.get $c))
                (call $posComponentAt (local.get $positionsOutBase) (local.get $a) (local.get $c)))
              (local.get $t))))
        (local.set $c (i32.add (local.get $c) (i32.const 1)))
        (br $posContinue)))
    (local.set $varyingAddr (call $scratchVaryingAddr (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $outCount)))
    (local.set $c (i32.const 0))
    (block $varyingBreak
      (loop $varyingContinue
        (br_if $varyingBreak (i32.ge_s (local.get $c) (local.get $wholeRecordComponents)))
        (f64.store
          (i32.add (local.get $varyingAddr) (i32.mul (local.get $c) (i32.const 8)))
          (f64.add
            (call $varyingComponentAt (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $a) (i32.const 0) (local.get $c))
            (f64.mul
              (f64.sub
                (call $varyingComponentAt (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $b) (i32.const 0) (local.get $c))
                (call $varyingComponentAt (local.get $varyingsOutBase) (local.get $varyingBytes) (local.get $a) (i32.const 0) (local.get $c)))
              (local.get $t))))
        (local.set $c (i32.add (local.get $c) (i32.const 1)))
        (br $varyingContinue)))
    (i32.add (local.get $outCount) (i32.const 1)))

  (func $clipEdge
    (param $positionsOutBase i32) (param $varyingsOutBase i32) (param $varyingBytes i32)
    (param $clipScratchBase i32) (param $wholeRecordComponents i32)
    (param $a i32) (param $b i32) (param $outCount i32) (result i32)
    (local $wA f64) (local $wB f64) (local $aIn i32) (local $bIn i32)
    (local.set $wA (call $posComponentAt (local.get $positionsOutBase) (local.get $a) (i32.const 3)))
    (local.set $wB (call $posComponentAt (local.get $positionsOutBase) (local.get $b) (i32.const 3)))
    (local.set $aIn (f64.gt (local.get $wA) (f64.const 1e-5)))
    (local.set $bIn (f64.gt (local.get $wB) (f64.const 1e-5)))
    (if (local.get $aIn)
      (then
        (local.set $outCount
          (call $vertexToScratch (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes)
            (local.get $clipScratchBase) (local.get $a) (local.get $outCount)))))
    (if (i32.ne (local.get $aIn) (local.get $bIn))
      (then
        (local.set $outCount
          (call $interpVertexToScratch
            (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes)
            (local.get $clipScratchBase) (local.get $wholeRecordComponents)
            (local.get $a) (local.get $b)
            (f64.div (f64.sub (f64.const 1e-5) (local.get $wA)) (f64.sub (local.get $wB) (local.get $wA)))
            (local.get $outCount)))))
    (local.get $outCount))

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
  (func $screenX (param $clippedPositionsOutBase i32) (param $widthF f64) (param $vertexIndex i32) (result f64)
    (f64.mul
      (f64.add
        (f64.mul
          (f64.div
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
            (f64.div
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
    (param $b0 f64) (param $invW0 f64) (param $b1 f64) (param $invW1 f64) (param $b2 f64) (param $invW2 f64) (param $invW f64)
    (local $d i32) (local $entry i32) (local $recordOffset i32) (local $fragmentDestAddress i32)
    (local $numComponents i32) (local $c i32)
    (local.set $d (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $d) (local.get $varyingDescCount)))
        (local.set $entry (call $descAddr (local.get $varyingDescBase) (local.get $d) (i32.const 16)))
        (local.set $recordOffset (i32.load (local.get $entry)))
        (local.set $fragmentDestAddress (i32.load (i32.add (local.get $entry) (i32.const 8))))
        (local.set $numComponents (i32.div_s (i32.load (i32.add (local.get $entry) (i32.const 12))) (i32.const 8)))
        (local.set $c (i32.const 0))
        (block $innerBreak
          (loop $innerContinue
            (br_if $innerBreak (i32.ge_s (local.get $c) (local.get $numComponents)))
            (f64.store
              (i32.add (local.get $fragmentDestAddress) (i32.mul (local.get $c) (i32.const 8)))
              (f64.div
                (f64.add
                  (f64.add
                    (f64.mul (f64.mul (local.get $b0) (local.get $invW0))
                      (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t0) (local.get $recordOffset) (local.get $c)))
                    (f64.mul (f64.mul (local.get $b1) (local.get $invW1))
                      (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t1) (local.get $recordOffset) (local.get $c))))
                  (f64.mul (f64.mul (local.get $b2) (local.get $invW2))
                    (call $varyingComponentAt (local.get $clippedVaryingsOutBase) (local.get $varyingBytes) (local.get $t2) (local.get $recordOffset) (local.get $c))))
                (local.get $invW)))
            (local.set $c (i32.add (local.get $c) (i32.const 1)))
            (br $innerContinue)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  ;; ---- rasterize: vertex pass, clip pass, triangle pass, in order ----
  (func $rasterize
    (param $vertexCount i32) (param $attrSrcBase i32) (param $attrStrideBytes i32)
    (param $attrDescBase i32) (param $attrDescCount i32) (param $vertexPositionAddress i32)
    (param $positionsOutBase i32) (param $width i32) (param $height i32)
    (param $fragmentValueAddress i32) (param $outputBase i32) (param $varyingBytes i32)
    (param $varyingDescBase i32) (param $varyingDescCount i32) (param $varyingsOutBase i32)
    (param $clipScratchBase i32) (param $clippedPositionsOutBase i32) (param $clippedVaryingsOutBase i32)
    (param $depthBufferBase i32)

    (local $i i32) (local $t i32) (local $t1 i32) (local $t2 i32)
    (local $widthF f64) (local $heightF f64)
    (local $wholeRecordComponents i32)
    (local $outCount i32) (local $clippedVertexCount i32)
    (local $s0x f64) (local $s0y f64) (local $s1x f64) (local $s1y f64) (local $s2x f64) (local $s2y f64)
    (local $w0 f64) (local $w1 f64) (local $w2 f64)
    (local $invW0 f64) (local $invW1 f64) (local $invW2 f64)
    (local $depth0 f64) (local $depth1 f64) (local $depth2 f64)
    (local $area f64)
    (local $minX i32) (local $maxX i32) (local $minY i32) (local $maxY i32)
    (local $x i32) (local $y i32) (local $px f64) (local $py f64) (local $pixelIndex i32)
    (local $e0 f64) (local $e1 f64) (local $e2 f64)
    (local $b0 f64) (local $b1 f64) (local $b2 f64) (local $invW f64)
    (local $pixelDepth f64) (local $depthAddr i32)

    (local.set $widthF (f64.convert_i32_s (local.get $width)))
    (local.set $heightF (f64.convert_i32_s (local.get $height)))
    (local.set $wholeRecordComponents (i32.div_s (local.get $varyingBytes) (i32.const 8)))

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

    ;; ---- clip pass ----
    (local.set $clippedVertexCount (i32.const 0))
    (local.set $t (i32.const 0))
    (block $clipBreak
      (loop $clipContinue
        (local.set $t1 (i32.add (local.get $t) (i32.const 1)))
        (local.set $t2 (i32.add (local.get $t) (i32.const 2)))
        (br_if $clipBreak (i32.ge_s (local.get $t2) (local.get $vertexCount)))
        (local.set $outCount (i32.const 0))
        (local.set $outCount
          (call $clipEdge (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes)
            (local.get $clipScratchBase) (local.get $wholeRecordComponents) (local.get $t) (local.get $t1) (local.get $outCount)))
        (local.set $outCount
          (call $clipEdge (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes)
            (local.get $clipScratchBase) (local.get $wholeRecordComponents) (local.get $t1) (local.get $t2) (local.get $outCount)))
        (local.set $outCount
          (call $clipEdge (local.get $positionsOutBase) (local.get $varyingsOutBase) (local.get $varyingBytes)
            (local.get $clipScratchBase) (local.get $wholeRecordComponents) (local.get $t2) (local.get $t) (local.get $outCount)))
        (if (i32.ge_s (local.get $outCount) (i32.const 3))
          (then
            (local.set $clippedVertexCount
              (call $emitTriangleFromScratch (local.get $clipScratchBase) (local.get $varyingBytes)
                (local.get $clippedPositionsOutBase) (local.get $clippedVaryingsOutBase)
                (local.get $clippedVertexCount) (i32.const 0) (i32.const 1) (i32.const 2)))))
        (if (i32.eq (local.get $outCount) (i32.const 4))
          (then
            (local.set $clippedVertexCount
              (call $emitTriangleFromScratch (local.get $clipScratchBase) (local.get $varyingBytes)
                (local.get $clippedPositionsOutBase) (local.get $clippedVaryingsOutBase)
                (local.get $clippedVertexCount) (i32.const 0) (i32.const 2) (i32.const 3)))))
        (local.set $t (i32.add (local.get $t) (i32.const 3)))
        (br $clipContinue)))

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
          (local.set $invW0 (f64.div (f64.const 1) (local.get $w0)))
          (local.set $invW1 (f64.div (f64.const 1) (local.get $w1)))
          (local.set $invW2 (f64.div (f64.const 1) (local.get $w2)))
          (local.set $depth0 (f64.div (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t) (i32.const 2)) (local.get $w0)))
          (local.set $depth1 (f64.div (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t1) (i32.const 2)) (local.get $w1)))
          (local.set $depth2 (f64.div (call $posComponentAt (local.get $clippedPositionsOutBase) (local.get $t2) (i32.const 2)) (local.get $w2)))
          (local.set $area
            (f64.sub
              (f64.mul (f64.sub (local.get $s1x) (local.get $s0x)) (f64.sub (local.get $s2y) (local.get $s0y)))
              (f64.mul (f64.sub (local.get $s1y) (local.get $s0y)) (f64.sub (local.get $s2x) (local.get $s0x)))))
          (br_if $skipDegenerate (f64.eq (local.get $area) (f64.const 0)))

          (local.set $minX (i32.trunc_f64_s (f64.max (f64.floor (f64.min (f64.min (local.get $s0x) (local.get $s1x)) (local.get $s2x))) (f64.const 0))))
          (local.set $maxX (i32.trunc_f64_s (f64.min (f64.ceil (f64.max (f64.max (local.get $s0x) (local.get $s1x)) (local.get $s2x))) (f64.sub (local.get $widthF) (f64.const 1)))))
          (local.set $minY (i32.trunc_f64_s (f64.max (f64.floor (f64.min (f64.min (local.get $s0y) (local.get $s1y)) (local.get $s2y))) (f64.const 0))))
          (local.set $maxY (i32.trunc_f64_s (f64.min (f64.ceil (f64.max (f64.max (local.get $s0y) (local.get $s1y)) (local.get $s2y))) (f64.sub (local.get $heightF) (f64.const 1)))))

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
                  ;; covered iff all three edge functions agree on sign
                  (if
                    (i32.or
                      (i32.and (i32.and (f64.ge (local.get $e0) (f64.const 0)) (f64.ge (local.get $e1) (f64.const 0))) (f64.ge (local.get $e2) (f64.const 0)))
                      (i32.and (i32.and (f64.le (local.get $e0) (f64.const 0)) (f64.le (local.get $e1) (f64.const 0))) (f64.le (local.get $e2) (f64.const 0))))
                    (then
                      (local.set $b0 (f64.div (local.get $e0) (local.get $area)))
                      (local.set $b1 (f64.div (local.get $e1) (local.get $area)))
                      (local.set $b2 (f64.div (local.get $e2) (local.get $area)))
                      (local.set $invW
                        (f64.add
                          (f64.add (f64.mul (local.get $b0) (local.get $invW0)) (f64.mul (local.get $b1) (local.get $invW1)))
                          (f64.mul (local.get $b2) (local.get $invW2))))
                      (local.set $pixelDepth
                        (f64.add
                          (f64.add (f64.mul (local.get $b0) (local.get $depth0)) (f64.mul (local.get $b1) (local.get $depth1)))
                          (f64.mul (local.get $b2) (local.get $depth2))))
                      (local.set $depthAddr (i32.add (local.get $depthBufferBase) (i32.mul (local.get $pixelIndex) (i32.const 8))))
                      ;; LEQUAL depth test: pre-clear depthBufferBase to a large value before the first draw
                      (if (f64.le (local.get $pixelDepth) (f64.load (local.get $depthAddr)))
                        (then
                          (f64.store (local.get $depthAddr) (local.get $pixelDepth))
                          (call $interpolateVaryings
                            (local.get $varyingDescBase) (local.get $varyingDescCount)
                            (local.get $clippedVaryingsOutBase) (local.get $varyingBytes)
                            (local.get $t) (local.get $t1) (local.get $t2)
                            (local.get $b0) (local.get $invW0) (local.get $b1) (local.get $invW1) (local.get $b2) (local.get $invW2) (local.get $invW))
                          (call $fragmentMain)
                          (call $byteCopy
                            (i32.add (local.get $outputBase) (i32.mul (local.get $pixelIndex) (i32.const 32)))
                            (local.get $fragmentValueAddress)
                            (i32.const 32))))))
                  (local.set $x (i32.add (local.get $x) (i32.const 1)))
                  (br $xContinue)))
              (local.set $y (i32.add (local.get $y) (i32.const 1)))
              (br $yContinue))))

        (local.set $t (i32.add (local.get $t) (i32.const 3)))
        (br $triContinue))))

  (export "rasterize" (func $rasterize)))
