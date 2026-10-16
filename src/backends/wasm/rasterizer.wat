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

;; ---- the lines of a triangle, after the 18 slots of the clip scratch; see lines.ts in the JS
  ;; backend, which computes them the same way. From $lines: its vertices, scaled, at +0, 4 f64
  ;; each; the exponent each was scaled down by at +96, an i32 each; each line's three
  ;; coefficients as four exact terms at +112, (3e + k) * 32; each coefficient rounded at +400,
  ;; and its size at +472; whether each line owns its centres at +544, an i32 each; the weight
  ;; of each vertex by mask at +560, mask * 24 + e * 8; at the pixel being tested, each line's
  ;; value at +752, then 24 bytes unused; the length of each line's expansion at +800 and the
  ;; expansions at +816, 256 bytes each; terms to sum at +1584, and their sum at +4144, 320 f64
  ;; each; the fragment's weights, their sum and its depth at +6704; and the plane of depth at
  ;; +6744: z's three coefficients along the ray, then the triangle's determinant; and at +6776,
  ;; per line, the part of its value and of its terms' sizes that a row of pixels shares, 16 bytes
  ;; each. ----
  (global $lines (mut i32) (i32.const 0))
  (global $clipScratch (mut i32) (i32.const 0))
  (global $varyingBytes (mut i32) (i32.const 0))
  (global $varyingDescBase (mut i32) (i32.const 0))
  (global $varyingDescCount (mut i32) (i32.const 0))
  (global $width (mut i32) (i32.const 0))
  (global $height (mut i32) (i32.const 0))
  (global $depthBufferBase (mut i32) (i32.const 0))
  (global $outputBase (mut i32) (i32.const 0))
  (global $fragmentValueAddress (mut i32) (i32.const 0))
  (global $writesColour (mut i32) (i32.const 0))
  (global $fragCoordAddress (mut i32) (i32.const 0))
  (global $readsFragCoord (mut i32) (i32.const 0))
  (global $fragDepthAddress (mut i32) (i32.const 0))
  (global $usesDepth (mut i32) (i32.const 0))
  (global $writesDepth (mut i32) (i32.const 0))
  (global $discardAddress (mut i32) (i32.const 0))
  (global $mayDiscard (mut i32) (i32.const 0))
  ;; The sign of the screen area of a triangle the draw culls, or 0 to cull none.
  (global $cullSign (mut f64) (f64.const 0))
  ;; Where in the clip scratch the polygon the last clip left starts.
  (global $clipped (mut i32) (i32.const 0))

  (func $linesBase (param $clipScratchBase i32) (param $varyingBytes i32) (result i32)
    (call $scratchVertexAddr (local.get $clipScratchBase) (local.get $varyingBytes) (i32.const 18)))

  ;; The exponent of the f64 at $addr, finite and not 0: floor(log2(|x|)), subnormals included.
  (func $exponentOf (param $addr i32) (result i32)
    (local $biased i32)
    (local.set $biased
      (i32.wrap_i64 (i64.and (i64.shr_u (i64.reinterpret_f64 (f64.load (local.get $addr))) (i64.const 52)) (i64.const 0x7ff))))
    (if (result i32) (local.get $biased)
      (then (i32.sub (local.get $biased) (i32.const 1023)))
      (else
        (i32.sub
          (i32.wrap_i64 (i64.and (i64.shr_u (i64.reinterpret_f64 (f64.mul (f64.load (local.get $addr)) (f64.const 0x1p64))) (i64.const 52)) (i64.const 0x7ff)))
          (i32.const 1087)))))

  ;; Multiplies the f64 at $addr by 2^$n, in steps that each power of two an f64 holds.
  (func $multiplyByPowerOfTwo (param $addr i32) (param $n i32)
    (block $done
      (loop $big
        (br_if $done (i32.le_s (local.get $n) (i32.const 1000)))
        (f64.store (local.get $addr) (f64.mul (f64.load (local.get $addr)) (f64.const 0x1p1000)))
        (local.set $n (i32.sub (local.get $n) (i32.const 1000)))
        (br $big)))
    (block $done
      (loop $small
        (br_if $done (i32.ge_s (local.get $n) (i32.const -1000)))
        (f64.store (local.get $addr) (f64.mul (f64.load (local.get $addr)) (f64.const 0x1p-1000)))
        (local.set $n (i32.add (local.get $n) (i32.const 1000)))
        (br $small)))
    (f64.store (local.get $addr)
      (f64.mul (f64.load (local.get $addr))
        (f64.reinterpret_i64 (i64.shl (i64.extend_i32_s (i32.add (local.get $n) (i32.const 1023))) (i64.const 52))))))

  ;; Writes $a times $b at $into, and the rounding of that product at $into + 8, so the two sum
  ;; to it exactly: Dekker's product, with the splitter 2^27 + 1.
  (func $product (param $a f64) (param $b f64) (param $into i32)
    (local $p f64) (local $c f64) (local $aHigh f64) (local $aLow f64) (local $bHigh f64) (local $bLow f64)
    (local.set $p (f64.mul (local.get $a) (local.get $b)))
    (local.set $c (f64.mul (f64.const 134217729) (local.get $a)))
    (local.set $aHigh (f64.sub (local.get $c) (f64.sub (local.get $c) (local.get $a))))
    (local.set $aLow (f64.sub (local.get $a) (local.get $aHigh)))
    (local.set $c (f64.mul (f64.const 134217729) (local.get $b)))
    (local.set $bHigh (f64.sub (local.get $c) (f64.sub (local.get $c) (local.get $b))))
    (local.set $bLow (f64.sub (local.get $b) (local.get $bHigh)))
    (f64.store (local.get $into) (local.get $p))
    (f64.store offset=8 (local.get $into)
      (f64.sub (f64.mul (local.get $aLow) (local.get $bLow))
        (f64.sub
          (f64.sub
            (f64.sub (local.get $p) (f64.mul (local.get $aHigh) (local.get $bHigh)))
            (f64.mul (local.get $aLow) (local.get $bHigh)))
          (f64.mul (local.get $aHigh) (local.get $bLow))))))

  ;; Adds $count f64 terms from $terms exactly into an empty expansion at $sum, and gives its
  ;; length: components smallest first, none of them 0 but a lone one, the last with the sign.
  (func $addExactly (param $terms i32) (param $count i32) (param $sum i32) (result i32)
    (local $length i32) (local $t i32) (local $i i32) (local $kept i32)
    (local $q f64) (local $e f64) (local $s f64) (local $bVirtual f64) (local $error f64)
    (block $termsDone
      (loop $nextTerm
        (br_if $termsDone (i32.ge_s (local.get $t) (local.get $count)))
        (local.set $q (f64.load (i32.add (local.get $terms) (i32.shl (local.get $t) (i32.const 3)))))
        (local.set $t (i32.add (local.get $t) (i32.const 1)))
        (br_if $nextTerm (f64.eq (local.get $q) (f64.const 0)))
        (local.set $kept (i32.const 0))
        (local.set $i (i32.const 0))
        (block $sumDone
          (loop $nextComponent
            (br_if $sumDone (i32.ge_s (local.get $i) (local.get $length)))
            (local.set $e (f64.load (i32.add (local.get $sum) (i32.shl (local.get $i) (i32.const 3)))))
            (local.set $s (f64.add (local.get $q) (local.get $e)))
            (local.set $bVirtual (f64.sub (local.get $s) (local.get $q)))
            (local.set $error
              (f64.add (f64.sub (local.get $q) (f64.sub (local.get $s) (local.get $bVirtual)))
                (f64.sub (local.get $e) (local.get $bVirtual))))
            (local.set $q (local.get $s))
            (if (f64.ne (local.get $error) (f64.const 0))
              (then
                (f64.store (i32.add (local.get $sum) (i32.shl (local.get $kept) (i32.const 3))) (local.get $error))
                (local.set $kept (i32.add (local.get $kept) (i32.const 1)))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $nextComponent)))
        (if (i32.or (f64.ne (local.get $q) (f64.const 0)) (i32.eqz (local.get $kept)))
          (then
            (f64.store (i32.add (local.get $sum) (i32.shl (local.get $kept) (i32.const 3))) (local.get $q))
            (local.set $kept (i32.add (local.get $kept) (i32.const 1)))))
        (local.set $length (local.get $kept))
        (br $nextTerm)))
    (local.get $length))

  ;; The sum of the expansion of $length components at $sum, as a float: within a few roundings of it.
  (func $estimateOf (param $sum i32) (param $length i32) (result f64)
    (local $i i32) (local $estimate f64)
    (block $done
      (loop $next
        (br_if $done (i32.ge_s (local.get $i) (local.get $length)))
        (local.set $estimate (f64.add (local.get $estimate) (f64.load (i32.add (local.get $sum) (i32.shl (local.get $i) (i32.const 3))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next)))
    (local.get $estimate))

  ;; The sign of the expansion of $length components at $sum.
  (func $signOf (param $sum i32) (param $length i32) (result i32)
    (local $top f64)
    (if (i32.eqz (local.get $length)) (then (return (i32.const 0))))
    (local.set $top (f64.load (i32.add (local.get $sum) (i32.shl (i32.sub (local.get $length) (i32.const 1)) (i32.const 3)))))
    (i32.sub (f64.gt (local.get $top) (f64.const 0)) (f64.lt (local.get $top) (f64.const 0))))

  ;; The sign of the sum of the four terms of the coefficient whose terms start at $at.
  (func $signOfTerms (param $at i32) (result i32)
    (local $l i32)
    (local.set $l (global.get $lines))
    (call $byteCopy (i32.add (local.get $l) (i32.const 1584)) (local.get $at) (i32.const 32))
    (call $signOf (i32.add (local.get $l) (i32.const 4144))
      (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 4) (i32.add (local.get $l) (i32.const 4144)))))

  ;; Writes the triangle whose positions are at $p0, $p1 and $p2 into the lines block, each vertex
  ;; scaled by a power of two that brings its largest component from 1 to 2, with the vertex $centre,
  ;; if not -1, at the centre of the target, as lines.ts's placeVertices does. A vertex at vec4(0)
  ;; stays 0, with a scale of 1. Gives whether one was.
  (func $placeVertices (param $p0 i32) (param $p1 i32) (param $p2 i32) (param $centre i32) (result i32)
    (local $l i32) (local $k i32) (local $c i32) (local $p i32) (local $o i32) (local $exponent i32) (local $least f64)
    (local $zero i32)
    (local.set $l (global.get $lines))
    (block $verticesDone
      (loop $nextVertex
        (br_if $verticesDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $p (select (local.get $p0) (select (local.get $p1) (local.get $p2) (i32.eq (local.get $k) (i32.const 1))) (i32.eqz (local.get $k))))
        (local.set $o (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 5))))
        (if (i32.eq (local.get $k) (local.get $centre))
          (then
            (f64.store (local.get $o) (f64.const 0))
            (f64.store offset=8 (local.get $o) (f64.const 0))
            (f64.store offset=16 (local.get $o) (f64.const 0))
            (f64.store offset=24 (local.get $o) (f64.const 1))
            (i32.store offset=96 (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 2))) (i32.const 0)))
          (else
            (f64.store offset=1584 (local.get $l)
              (f64.max
                (f64.max (f64.abs (f64.load (local.get $p))) (f64.abs (f64.load offset=8 (local.get $p))))
                (f64.max (f64.abs (f64.load offset=16 (local.get $p))) (f64.abs (f64.load offset=24 (local.get $p))))))
            (if (f64.eq (f64.load offset=1584 (local.get $l)) (f64.const 0))
              (then
                (f64.store (local.get $o) (f64.const 0))
                (f64.store offset=8 (local.get $o) (f64.const 0))
                (f64.store offset=16 (local.get $o) (f64.const 0))
                (f64.store offset=24 (local.get $o) (f64.const 0))
                (i32.store offset=96 (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 2))) (i32.const 0))
                (local.set $zero (i32.const 1))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $nextVertex)))
            (local.set $exponent (call $exponentOf (i32.add (local.get $l) (i32.const 1584))))
            (i32.store offset=96 (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 2))) (local.get $exponent))
            (local.set $c (i32.const 0))
            (block $scaled
              (loop $nextComponent
                (br_if $scaled (i32.ge_s (local.get $c) (i32.const 4)))
                (f64.store (i32.add (local.get $o) (i32.shl (local.get $c) (i32.const 3)))
                  (f64.load (i32.add (local.get $p) (i32.shl (local.get $c) (i32.const 3)))))
                (call $multiplyByPowerOfTwo (i32.add (local.get $o) (i32.shl (local.get $c) (i32.const 3))) (i32.sub (i32.const 0) (local.get $exponent)))
                (local.set $c (i32.add (local.get $c) (i32.const 1)))
                (br $nextComponent)))
            ;; The largest component now lies from 1 to 2, so the least that counts is 2^-300 of it.
            (local.set $least
              (f64.mul
                (f64.max
                  (f64.max (f64.abs (f64.load (local.get $o))) (f64.abs (f64.load offset=8 (local.get $o))))
                  (f64.max (f64.abs (f64.load offset=16 (local.get $o))) (f64.abs (f64.load offset=24 (local.get $o)))))
                (f64.const 0x1p-300)))
            (local.set $c (i32.const 0))
            (block $flushed
              (loop $nextFlush
                (br_if $flushed (i32.ge_s (local.get $c) (i32.const 4)))
                (if (f64.lt (f64.abs (f64.load (i32.add (local.get $o) (i32.shl (local.get $c) (i32.const 3))))) (local.get $least))
                  (then (f64.store (i32.add (local.get $o) (i32.shl (local.get $c) (i32.const 3))) (f64.const 0))))
                (local.set $c (i32.add (local.get $c) (i32.const 1)))
                (br $nextFlush)))))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextVertex)))
    (local.get $zero))

  ;; Writes each line's coefficients as four exact terms, each rounded, and the size of its terms,
  ;; as lines.ts's measureLines does.
  (func $measureLines
    (local $l i32) (local $e i32) (local $k i32) (local $i i32) (local $j i32) (local $u i32) (local $v i32) (local $t i32)
    (local $q i32)
    (local.set $l (global.get $lines))
    ;; Line e, opposite vertex e, through vertices i and j: a = yi wj - wi yj, b = wi xj - xi wj, c = xi yj - yi xj.
    (local.set $e (i32.const 0))
    (block $linesDone
      (loop $nextLine
        (br_if $linesDone (i32.ge_s (local.get $e) (i32.const 3)))
        (local.set $i (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $e) (i32.const 1)) (i32.const 3)) (i32.const 5))))
        (local.set $j (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $e) (i32.const 2)) (i32.const 3)) (i32.const 5))))
        (local.set $k (i32.const 0))
        (block $coefficientsDone
          (loop $nextCoefficient
            (br_if $coefficientsDone (i32.ge_s (local.get $k) (i32.const 3)))
            ;; The two components each product takes, of x (0), y (8) and w (24), as byte offsets.
            (local.set $u (select (i32.const 8) (select (i32.const 24) (i32.const 0) (i32.eq (local.get $k) (i32.const 1))) (i32.eqz (local.get $k))))
            (local.set $v (select (i32.const 24) (select (i32.const 0) (i32.const 8) (i32.eq (local.get $k) (i32.const 1))) (i32.eqz (local.get $k))))
            (local.set $t (i32.add (i32.add (local.get $l) (i32.const 112))
              (i32.shl (i32.add (i32.mul (local.get $e) (i32.const 3)) (local.get $k)) (i32.const 5))))
            (call $product (f64.load (i32.add (local.get $i) (local.get $u))) (f64.load (i32.add (local.get $j) (local.get $v))) (local.get $t))
            (call $product (f64.load (i32.add (local.get $i) (local.get $v))) (f64.load (i32.add (local.get $j) (local.get $u))) (i32.add (local.get $t) (i32.const 16)))
            (f64.store offset=16 (local.get $t) (f64.neg (f64.load offset=16 (local.get $t))))
            (f64.store offset=24 (local.get $t) (f64.neg (f64.load offset=24 (local.get $t))))
            (local.set $q (i32.add (local.get $l) (i32.shl (i32.add (i32.mul (local.get $e) (i32.const 3)) (local.get $k)) (i32.const 3))))
            (f64.store offset=400 (local.get $q)
              (f64.add
                (f64.add (f64.load (local.get $t)) (f64.load offset=16 (local.get $t)))
                (f64.add (f64.load offset=8 (local.get $t)) (f64.load offset=24 (local.get $t)))))
            (f64.store offset=472 (local.get $q)
              (f64.add (f64.abs (f64.load (local.get $t))) (f64.abs (f64.load offset=16 (local.get $t)))))
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br $nextCoefficient)))
        (local.set $e (i32.add (local.get $e) (i32.const 1)))
        (br $nextLine))))

  ;; Prepares the lines of the triangle whose positions are at $p0, $p1 and $p2, with the vertex
  ;; $centre, if not -1, at the centre of the target in place of vec4(0), as lines.ts's
  ;; prepareLines does. Gives the triangle's sign, 0 when its vertices lie in one plane with the eye.
  (func $prepareLines (param $p0 i32) (param $p1 i32) (param $p2 i32) (param $centre i32) (result i32)
    (local $l i32) (local $k i32) (local $e i32) (local $q i32) (local $n i32)
    (local $orientation i32) (local $a i32) (local $b i32) (local $mask i32) (local $leastExponent i32) (local $r i32)
    (local $length i32)
    (local.set $l (global.get $lines))
    (if (call $placeVertices (local.get $p0) (local.get $p1) (local.get $p2) (local.get $centre)) (then (return (i32.const 0))))
    (call $measureLines)
    ;; The sign of the triangle: line 0 at vertex 0, exactly.
    (local.set $n (i32.const 0))
    (local.set $k (i32.const 0))
    (block $termsDone
      (loop $nextK
        (br_if $termsDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $q (i32.const 0))
        (block $qDone
          (loop $nextQ
            (br_if $qDone (i32.ge_s (local.get $q) (i32.const 4)))
            (call $product
              (f64.load (i32.add (i32.add (local.get $l) (i32.const 112)) (i32.shl (i32.add (i32.shl (local.get $k) (i32.const 2)) (local.get $q)) (i32.const 3))))
              (f64.load (i32.add (local.get $l) (select (i32.const 24) (i32.shl (local.get $k) (i32.const 3)) (i32.eq (local.get $k) (i32.const 2)))))
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
            (local.set $n (i32.add (local.get $n) (i32.const 2)))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextQ)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextK)))
    (local.set $length (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n) (i32.add (local.get $l) (i32.const 4144))))
    (local.set $orientation (call $signOf (i32.add (local.get $l) (i32.const 4144)) (local.get $length)))
    (if (i32.eqz (local.get $orientation)) (then (return (i32.const 0))))
    (f64.store offset=6768 (local.get $l) (call $estimateOf (i32.add (local.get $l) (i32.const 4144)) (local.get $length)))
    ;; The point's z along the ray is the sum of each vertex's z times its line: each coefficient summed exactly once.
    (local.set $k (i32.const 0))
    (block $planeDone
      (loop $nextPlane
        (br_if $planeDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $n (i32.const 0))
        (local.set $e (i32.const 0))
        (block $planeLinesDone
          (loop $nextPlaneLine
            (br_if $planeLinesDone (i32.ge_s (local.get $e) (i32.const 3)))
            (local.set $q (i32.const 0))
            (block $planeTermsDone
              (loop $nextPlaneTerm
                (br_if $planeTermsDone (i32.ge_s (local.get $q) (i32.const 4)))
                (call $product
                  (f64.load (i32.add (i32.add (local.get $l) (i32.const 112))
                    (i32.shl (i32.add (i32.shl (i32.add (i32.mul (local.get $e) (i32.const 3)) (local.get $k)) (i32.const 2)) (local.get $q)) (i32.const 3))))
                  (f64.load offset=16 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 5))))
                  (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
                (local.set $n (i32.add (local.get $n) (i32.const 2)))
                (local.set $q (i32.add (local.get $q) (i32.const 1)))
                (br $nextPlaneTerm)))
            (local.set $e (i32.add (local.get $e) (i32.const 1)))
            (br $nextPlaneLine)))
        (f64.store offset=6744 (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 3)))
          (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
            (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n) (i32.add (local.get $l) (i32.const 4144)))))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextPlane)))
    ;; An edge owns its centres when its inside lies to its right on the screen, or below it when horizontal.
    (local.set $e (i32.const 0))
    (block $ownsDone
      (loop $nextOwns
        (br_if $ownsDone (i32.ge_s (local.get $e) (i32.const 3)))
        (local.set $a (i32.mul (call $signOfTerms (i32.add (i32.add (local.get $l) (i32.const 112)) (i32.mul (local.get $e) (i32.const 96)))) (local.get $orientation)))
        (local.set $b (i32.mul (i32.sub (i32.const 0) (call $signOfTerms (i32.add (i32.add (local.get $l) (i32.const 144)) (i32.mul (local.get $e) (i32.const 96))))) (local.get $orientation)))
        (i32.store offset=544 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))
          (i32.or (i32.gt_s (local.get $a) (i32.const 0)) (i32.and (i32.eqz (local.get $a)) (i32.gt_s (local.get $b) (i32.const 0)))))
        (local.set $e (i32.add (local.get $e) (i32.const 1)))
        (br $nextOwns)))
    ;; The weight of a vertex for each unit of its line is its scale over the greatest scale among the
    ;; vertices whose lines are not 0 at the pixel: a smaller vertex, of a smaller w, weighs more.
    (local.set $mask (i32.const 1))
    (block $masksDone
      (loop $nextMask
        (br_if $masksDone (i32.ge_s (local.get $mask) (i32.const 8)))
        (local.set $leastExponent (i32.const 0x7fffffff))
        (local.set $e (i32.const 0))
        (block $leastDone
          (loop $nextLeast
            (br_if $leastDone (i32.ge_s (local.get $e) (i32.const 3)))
            (if (i32.and (local.get $mask) (i32.shl (i32.const 1) (local.get $e)))
              (then
                (if (i32.lt_s (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))) (local.get $leastExponent))
                  (then (local.set $leastExponent (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))))))))
            (local.set $e (i32.add (local.get $e) (i32.const 1)))
            (br $nextLeast)))
        (local.set $e (i32.const 0))
        (block $ratiosDone
          (loop $nextRatio
            (br_if $ratiosDone (i32.ge_s (local.get $e) (i32.const 3)))
            (local.set $r (i32.add (i32.add (local.get $l) (i32.const 560)) (i32.shl (i32.add (i32.mul (local.get $mask) (i32.const 3)) (local.get $e)) (i32.const 3))))
            (if (i32.and (local.get $mask) (i32.shl (i32.const 1) (local.get $e)))
              (then
                (f64.store (local.get $r) (f64.const 1))
                (call $multiplyByPowerOfTwo (local.get $r)
                  (select
                    (i32.sub (local.get $leastExponent) (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))))
                    (i32.const -1100)
                    (i32.gt_s
                      (i32.sub (local.get $leastExponent) (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))))
                      (i32.const -1100)))))
              (else (f64.store (local.get $r) (f64.const 0))))
            (local.set $e (i32.add (local.get $e) (i32.const 1)))
            (br $nextRatio)))
        (local.set $mask (i32.add (local.get $mask) (i32.const 1)))
        (br $nextMask)))
    (local.get $orientation))

  ;; Writes the exact expansion of line $e's value at the ray ($X, $Y, $W) and gives its length.
  (func $expandLine (param $e i32) (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $k i32) (local $q i32) (local $n i32) (local $length i32) (local $whole f64)
    (local.set $l (global.get $lines))
    (block $kDone
      (loop $nextK
        (br_if $kDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $whole (select (local.get $X) (select (local.get $Y) (local.get $W) (i32.eq (local.get $k) (i32.const 1))) (i32.eqz (local.get $k))))
        (local.set $q (i32.const 0))
        (block $qDone
          (loop $nextQ
            (br_if $qDone (i32.ge_s (local.get $q) (i32.const 4)))
            (call $product
              (f64.load (i32.add (i32.add (local.get $l) (i32.const 112))
                (i32.shl (i32.add (i32.shl (i32.add (i32.mul (local.get $e) (i32.const 3)) (local.get $k)) (i32.const 2)) (local.get $q)) (i32.const 3))))
              (local.get $whole)
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
            (local.set $n (i32.add (local.get $n) (i32.const 2)))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextQ)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextK)))
    (local.set $length
      (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n)
        (i32.add (i32.add (local.get $l) (i32.const 816)) (i32.shl (local.get $e) (i32.const 8)))))
    (i32.store offset=800 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2))) (local.get $length))
    (local.get $length))

;; Writes the part of each line's float value and of its terms' sizes that the row of pixels at
  ;; $Y shares, as lines.ts's prepareRow does.
  (func $prepareRow (param $Y f64) (param $W f64)
    (local $l i32) (local $e i32) (local $k i32)
    (local.set $l (global.get $lines))
    (block $done
      (loop $next
        (br_if $done (i32.ge_s (local.get $e) (i32.const 3)))
        (local.set $k (i32.add (local.get $l) (i32.mul (local.get $e) (i32.const 24))))
        (f64.store offset=6776 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 4)))
          (f64.add (f64.mul (f64.load offset=408 (local.get $k)) (local.get $Y)) (f64.mul (f64.load offset=416 (local.get $k)) (local.get $W))))
        (f64.store offset=6784 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 4)))
          (f64.add (f64.mul (f64.load offset=480 (local.get $k)) (f64.abs (local.get $Y))) (f64.mul (f64.load offset=488 (local.get $k)) (local.get $W))))
        (local.set $e (i32.add (local.get $e) (i32.const 1)))
        (br $next))))

  ;; Tests line $e at the ray ($X, $Y, $W), in the row $prepareRow prepared, as lines.ts's lineSign
  ;; does: writes its value and gives its sign, exactly. The float value is kept when it is more than
  ;; 2^-16 of its terms' sizes, 2^32 times its bound of rounding; any other is summed exactly, and its
  ;; value is the expansion's float sum.
  (func $lineSign (param $e i32) (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $row i32) (local $value f64) (local $bound f64) (local $length i32) (local $expansion i32)
    (local.set $l (global.get $lines))
    (local.set $row (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 4))))
    (local.set $value
      (f64.add (f64.mul (f64.load offset=400 (i32.add (local.get $l) (i32.mul (local.get $e) (i32.const 24)))) (local.get $X))
        (f64.load offset=6776 (local.get $row))))
    (local.set $bound
      (f64.mul (f64.const 0x1p-16)
        (f64.add (f64.mul (f64.load offset=472 (i32.add (local.get $l) (i32.mul (local.get $e) (i32.const 24)))) (f64.abs (local.get $X)))
          (f64.load offset=6784 (local.get $row)))))
    (if (i32.or (f64.gt (local.get $value) (local.get $bound)) (f64.lt (local.get $value) (f64.neg (local.get $bound))))
      (then
        (f64.store offset=752 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 3))) (local.get $value))
        (return (select (i32.const 1) (i32.const -1) (f64.gt (local.get $value) (f64.const 0))))))
    (local.set $length (call $expandLine (local.get $e) (local.get $X) (local.get $Y) (local.get $W)))
    (local.set $expansion (i32.add (i32.add (local.get $l) (i32.const 816)) (i32.shl (local.get $e) (i32.const 8))))
    (f64.store offset=752 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 3)))
      (call $estimateOf (local.get $expansion) (local.get $length)))
    (call $signOf (local.get $expansion) (local.get $length)))

  ;; The exact sign, at the ray ($X, $Y, $W) whose lines $lineSign has tested, of the point's z
  ;; ($component 2) or its w minus its z (-1), as lines.ts's pointSign gives it.
  (func $pointSign (param $component i32) (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $z i32) (local $w i32) (local $minus f64)
    (local $c0 f64) (local $c1 f64) (local $c2 f64) (local $m0 f64) (local $m1 f64) (local $m2 f64) (local $value f64) (local $bound f64)
    (local.set $l (global.get $lines))
    (local.set $z (i32.shl (select (i32.const 2) (local.get $component) (i32.eq (local.get $component) (i32.const -1))) (i32.const 3)))
    (local.set $w (i32.shl (select (i32.const 3) (local.get $component) (i32.eq (local.get $component) (i32.const -1))) (i32.const 3)))
    (local.set $minus (select (f64.const 1) (f64.const 0) (i32.eq (local.get $component) (i32.const -1))))
    (local.set $c0 (f64.sub (f64.load (i32.add (local.get $l) (local.get $w))) (f64.mul (local.get $minus) (f64.load (i32.add (local.get $l) (local.get $z))))))
    (local.set $c1 (f64.sub (f64.load offset=32 (i32.add (local.get $l) (local.get $w))) (f64.mul (local.get $minus) (f64.load offset=32 (i32.add (local.get $l) (local.get $z))))))
    (local.set $c2 (f64.sub (f64.load offset=64 (i32.add (local.get $l) (local.get $w))) (f64.mul (local.get $minus) (f64.load offset=64 (i32.add (local.get $l) (local.get $z))))))
    (local.set $m0 (f64.add (f64.abs (f64.load (i32.add (local.get $l) (local.get $w)))) (f64.mul (local.get $minus) (f64.abs (f64.load (i32.add (local.get $l) (local.get $z)))))))
    (local.set $m1 (f64.add (f64.abs (f64.load offset=32 (i32.add (local.get $l) (local.get $w)))) (f64.mul (local.get $minus) (f64.abs (f64.load offset=32 (i32.add (local.get $l) (local.get $z)))))))
    (local.set $m2 (f64.add (f64.abs (f64.load offset=64 (i32.add (local.get $l) (local.get $w)))) (f64.mul (local.get $minus) (f64.abs (f64.load offset=64 (i32.add (local.get $l) (local.get $z)))))))
    (local.set $value
      (f64.add
        (f64.add (f64.mul (local.get $c0) (f64.load offset=752 (local.get $l))) (f64.mul (local.get $c1) (f64.load offset=760 (local.get $l))))
        (f64.mul (local.get $c2) (f64.load offset=768 (local.get $l)))))
    ;; Each line's bound of rounding is at least 2^-48 of its value too, which covers the rounding of this sum.
    (local.set $bound
      (f64.add
        (f64.add
          (f64.mul (local.get $m0) (f64.mul (f64.const 0x1p-48)
            (f64.add (f64.mul (f64.load offset=472 (local.get $l)) (f64.abs (local.get $X))) (f64.load offset=6784 (local.get $l)))))
          (f64.mul (local.get $m1) (f64.mul (f64.const 0x1p-48)
            (f64.add (f64.mul (f64.load offset=496 (local.get $l)) (f64.abs (local.get $X))) (f64.load offset=6800 (local.get $l))))))
        (f64.mul (local.get $m2) (f64.mul (f64.const 0x1p-48)
          (f64.add (f64.mul (f64.load offset=520 (local.get $l)) (f64.abs (local.get $X))) (f64.load offset=6816 (local.get $l)))))))
    (if (f64.gt (local.get $value) (local.get $bound)) (then (return (i32.const 1))))
    (if (f64.lt (local.get $value) (f64.neg (local.get $bound))) (then (return (i32.const -1))))
    ;; A bound of 0 leaves every component 0, as a line of a triangle that is not flat is not 0: so is the sum.
    (if (f64.eq (local.get $bound) (f64.const 0)) (then (return (i32.const 0))))
    (call $signOf (i32.add (local.get $l) (i32.const 4144))
      (call $expandPoint (local.get $component) (local.get $X) (local.get $Y) (local.get $W))))

  ;; Writes into the sum at +4144 the exact expansion, at the ray ($X, $Y, $W), of the point's z
  ;; ($component 2) or its w minus its z (-1), as lines.ts's expandPoint does, and gives its length.
  (func $expandPoint (param $component i32) (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $z i32) (local $w i32) (local $e i32) (local $i i32) (local $n i32) (local $length i32) (local $at i32)
    (local.set $l (global.get $lines))
    (local.set $z (i32.shl (select (i32.const 2) (local.get $component) (i32.eq (local.get $component) (i32.const -1))) (i32.const 3)))
    (local.set $w (i32.shl (select (i32.const 3) (local.get $component) (i32.eq (local.get $component) (i32.const -1))) (i32.const 3)))
    ;; Each line's expansion first, as an expansion takes the terms to sum for its own.
    (drop (call $expandLine (i32.const 0) (local.get $X) (local.get $Y) (local.get $W)))
    (drop (call $expandLine (i32.const 1) (local.get $X) (local.get $Y) (local.get $W)))
    (drop (call $expandLine (i32.const 2) (local.get $X) (local.get $Y) (local.get $W)))
    (block $linesDone
      (loop $nextLine
        (br_if $linesDone (i32.ge_s (local.get $e) (i32.const 3)))
        (local.set $length (i32.load offset=800 (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 2)))))
        (local.set $i (i32.const 0))
        (block $componentsDone
          (loop $nextComponent
            (br_if $componentsDone (i32.ge_s (local.get $i) (local.get $length)))
            (local.set $at (i32.add (i32.add (i32.add (local.get $l) (i32.const 816)) (i32.shl (local.get $e) (i32.const 8))) (i32.shl (local.get $i) (i32.const 3))))
            (call $product (f64.load (local.get $at))
              (f64.load (i32.add (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 5))) (local.get $w)))
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
            (local.set $n (i32.add (local.get $n) (i32.const 2)))
            (if (i32.eq (local.get $component) (i32.const -1))
              (then
                (call $product (f64.load (local.get $at))
                  (f64.neg (f64.load (i32.add (i32.add (local.get $l) (i32.shl (local.get $e) (i32.const 5))) (local.get $z))))
                  (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
                (local.set $n (i32.add (local.get $n) (i32.const 2)))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $nextComponent)))
        (local.set $e (i32.add (local.get $e) (i32.const 1)))
        (br $nextLine)))
    (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n) (i32.add (local.get $l) (i32.const 4144))))

  ;; The depth at the ray ($X, $Y, $W) whose lines $lineSign has tested, held to 0 to 1, as
  ;; lines.ts's depthAt gives it: the plane of depth's float value where its bound of rounding,
  ;; 2^-50 of its terms, is within 2^-32 of the point's w, and the point's z summed exactly elsewhere.
  (func $depthAt (param $X f64) (param $Y f64) (param $W f64) (result f64)
    (local $l i32) (local $x f64) (local $y f64) (local $w f64) (local $divisor f64) (local $z f64)
    (local.set $l (global.get $lines))
    (local.set $x (f64.mul (f64.load offset=6744 (local.get $l)) (local.get $X)))
    (local.set $y (f64.mul (f64.load offset=6752 (local.get $l)) (local.get $Y)))
    (local.set $w (f64.mul (f64.load offset=6760 (local.get $l)) (local.get $W)))
    (local.set $divisor (f64.mul (f64.load offset=6768 (local.get $l)) (local.get $W)))
    (local.set $z (f64.add (f64.add (local.get $x) (local.get $y)) (local.get $w)))
    (if (f64.gt
          (f64.mul (f64.const 0x1p-50) (f64.add (f64.add (f64.abs (local.get $x)) (f64.abs (local.get $y))) (f64.abs (local.get $w))))
          (f64.mul (f64.const 0x1p-32) (f64.abs (local.get $divisor))))
      (then
        (local.set $z
          (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
            (call $expandPoint (i32.const 2) (local.get $X) (local.get $Y) (local.get $W))))))
    (f64.min (f64.const 1) (f64.max (f64.const 0) (f64.div (local.get $z) (local.get $divisor)))))

  ;; ---- the triangle a triangle in one plane with the eye draws, as lines.ts's prepareWedge
  ;; measures it. From $lines: each side's coefficients as eight exact terms at +6824,
  ;; (3s + k) * 64; each rounded at +8744 and the size of its terms at +8984, (3s + k) * 8; whether
  ;; each side owns its centres at +9224, an i32 each; how many sides at +9264; the depth along the
  ;; ray at +9272: the four terms of dX, then of dY, each rounded, and c; and the weight of each
  ;; vertex at +9360. ----

  ;; Adds to coefficient $k of side $s the $count terms at $from, each times $sign, from term $slot
  ;; of its eight, as lines.ts's put does.
  (func $put (param $s i32) (param $k i32) (param $slot i32) (param $from i32) (param $count i32) (param $sign f64)
    (local $into i32) (local $q i32)
    (local.set $into
      (i32.add (i32.add (global.get $lines) (i32.const 6824))
        (i32.shl (i32.add (i32.shl (i32.add (i32.mul (local.get $s) (i32.const 3)) (local.get $k)) (i32.const 3)) (local.get $slot)) (i32.const 3))))
    (block $done
      (loop $next
        (br_if $done (i32.ge_s (local.get $q) (local.get $count)))
        (f64.store (i32.add (local.get $into) (i32.shl (local.get $q) (i32.const 3)))
          (f64.mul (f64.load (i32.add (local.get $from) (i32.shl (local.get $q) (i32.const 3)))) (local.get $sign)))
        (local.set $q (i32.add (local.get $q) (i32.const 1)))
        (br $next))))

  ;; Rounds the coefficients of side $s, sums the sizes of their terms and decides whether it owns its
  ;; centres, as lines.ts's closeSide does. Gives $s + 1, or $s for a side that is 0 everywhere, which
  ;; is cleared for the next.
  (func $closeSide (param $s i32) (result i32)
    (local $l i32) (local $k i32) (local $q i32) (local $at i32) (local $value f64) (local $size f64) (local $term f64)
    (local $sign i32) (local $x i32) (local $y i32) (local $w i32)
    (local.set $l (global.get $lines))
    (block $coefficientsDone
      (loop $nextCoefficient
        (br_if $coefficientsDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $at (i32.add (i32.add (local.get $l) (i32.const 6824))
          (i32.shl (i32.add (i32.mul (local.get $s) (i32.const 3)) (local.get $k)) (i32.const 6))))
        (local.set $value (f64.const 0))
        (local.set $size (f64.const 0))
        (local.set $q (i32.const 0))
        (block $termsDone
          (loop $nextTerm
            (br_if $termsDone (i32.ge_s (local.get $q) (i32.const 8)))
            (local.set $term (f64.load (i32.add (local.get $at) (i32.shl (local.get $q) (i32.const 3)))))
            (local.set $value (f64.add (local.get $value) (local.get $term)))
            (local.set $size (f64.add (local.get $size) (f64.abs (local.get $term))))
            (f64.store offset=1584 (i32.add (local.get $l) (i32.shl (local.get $q) (i32.const 3))) (local.get $term))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextTerm)))
        (f64.store offset=8744 (i32.add (local.get $l) (i32.shl (i32.add (i32.mul (local.get $s) (i32.const 3)) (local.get $k)) (i32.const 3))) (local.get $value))
        (f64.store offset=8984 (i32.add (local.get $l) (i32.shl (i32.add (i32.mul (local.get $s) (i32.const 3)) (local.get $k)) (i32.const 3))) (local.get $size))
        (local.set $sign
          (call $signOf (i32.add (local.get $l) (i32.const 4144))
            (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 8) (i32.add (local.get $l) (i32.const 4144)))))
        (if (i32.eqz (local.get $k)) (then (local.set $x (local.get $sign))))
        (if (i32.eq (local.get $k) (i32.const 1)) (then (local.set $y (local.get $sign))))
        (if (i32.eq (local.get $k) (i32.const 2)) (then (local.set $w (local.get $sign))))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextCoefficient)))
    (if (i32.eqz (i32.or (i32.or (local.get $x) (local.get $y)) (local.get $w)))
      (then
        (local.set $at (i32.add (i32.add (local.get $l) (i32.const 6824)) (i32.mul (local.get $s) (i32.const 192))))
        (local.set $q (i32.const 0))
        (block $cleared
          (loop $nextClear
            (br_if $cleared (i32.ge_s (local.get $q) (i32.const 24)))
            (f64.store (i32.add (local.get $at) (i32.shl (local.get $q) (i32.const 3))) (f64.const 0))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextClear)))
        (return (local.get $s))))
    ;; A side owns its centres when its inside lies to its right on the screen, or below it when horizontal.
    (i32.store offset=9224 (i32.add (local.get $l) (i32.shl (local.get $s) (i32.const 2)))
      (i32.or (i32.gt_s (local.get $x) (i32.const 0)) (i32.and (i32.eqz (local.get $x)) (i32.lt_s (local.get $y) (i32.const 0)))))
    (i32.add (local.get $s) (i32.const 1)))

  ;; Prepares the triangle whose positions are at $p0, $p1 and $p2, whose vertices lie in one plane
  ;; with the eye, as lines.ts's prepareWedge does: where it contains vec4(0), it draws the triangle
  ;; from the centre to the two ends of the segment the rest of its part inside the clip volume
  ;; projects to. Gives that triangle's winding in normalized device coordinates, or 0 when it draws nothing.
  (func $prepareWedge (param $p0 i32) (param $p1 i32) (param $p2 i32) (result i32)
    (local $l i32) (local $f i32) (local $q i32) (local $n i32) (local $e i32) (local $sign i32) (local $orientation i32)
    (local $i i32) (local $j i32) (local $d i32) (local $a i32) (local $b i32) (local $c i32) (local $o f64) (local $s i32)
    (local $k i32) (local $pj i32) (local $pk i32) (local $least i32) (local $exponent i32) (local $weight i32)
    (local.set $l (global.get $lines))
    (drop (call $placeVertices (local.get $p0) (local.get $p1) (local.get $p2) (i32.const -1)))
    (call $measureLines)
    (local.set $q (i32.const 0))
    (block $cleared
      (loop $nextClear
        (br_if $cleared (i32.ge_s (local.get $q) (i32.const 240)))
        (f64.store offset=6824 (i32.add (local.get $l) (i32.shl (local.get $q) (i32.const 3))) (f64.const 0))
        (local.set $q (i32.add (local.get $q) (i32.const 1)))
        (br $nextClear)))
    (local.set $e (i32.const -1))
    (block $weightsDone
      (loop $nextWeight
        (br_if $weightsDone (i32.ge_s (local.get $f) (i32.const 3)))
        (local.set $sign (call $signOfTerms (i32.add (i32.add (local.get $l) (i32.const 176)) (i32.mul (local.get $f) (i32.const 96)))))
        (if (local.get $sign)
          (then
            (if (i32.and (i32.ne (local.get $orientation) (i32.const 0)) (i32.ne (local.get $sign) (local.get $orientation)))
              (then (return (i32.const 0))))
            (local.set $orientation (local.get $sign))
            (if (i32.eq (local.get $e) (i32.const -1)) (then (local.set $e (local.get $f))))))
        (local.set $f (i32.add (local.get $f) (i32.const 1)))
        (br $nextWeight)))
    (if (i32.eqz (local.get $orientation)) (then (return (i32.const 0))))
    ;; The vertices' z by those weights, exactly: 0 where vec4(0) is the triangle's point of w 0.
    (local.set $f (i32.const 0))
    (block $zDone
      (loop $nextZ
        (br_if $zDone (i32.ge_s (local.get $f) (i32.const 3)))
        (local.set $q (i32.const 0))
        (block $qDone
          (loop $nextQ
            (br_if $qDone (i32.ge_s (local.get $q) (i32.const 4)))
            (call $product
              (f64.load (i32.add (i32.add (local.get $l) (i32.const 176))
                (i32.add (i32.mul (local.get $f) (i32.const 96)) (i32.shl (local.get $q) (i32.const 3)))))
              (f64.load offset=16 (i32.add (local.get $l) (i32.shl (local.get $f) (i32.const 5))))
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
            (local.set $n (i32.add (local.get $n) (i32.const 2)))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextQ)))
        (local.set $f (i32.add (local.get $f) (i32.const 1)))
        (br $nextZ)))
    (if (call $signOf (i32.add (local.get $l) (i32.const 4144))
          (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n) (i32.add (local.get $l) (i32.const 4144))))
      (then (return (i32.const 0))))
    ;; z along the ray, by the two vertices line e runs through: dX = yj zi - yi zj, dY = xi zj - xj zi.
    (local.set $i (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $e) (i32.const 1)) (i32.const 3)) (i32.const 5))))
    (local.set $j (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $e) (i32.const 2)) (i32.const 3)) (i32.const 5))))
    (local.set $d (i32.add (local.get $l) (i32.const 9272)))
    (call $product (f64.load offset=8 (local.get $j)) (f64.load offset=16 (local.get $i)) (local.get $d))
    (call $product (f64.load offset=8 (local.get $i)) (f64.load offset=16 (local.get $j)) (i32.add (local.get $d) (i32.const 16)))
    (call $product (f64.load (local.get $i)) (f64.load offset=16 (local.get $j)) (i32.add (local.get $d) (i32.const 32)))
    (call $product (f64.load (local.get $j)) (f64.load offset=16 (local.get $i)) (i32.add (local.get $d) (i32.const 48)))
    (f64.store offset=16 (local.get $d) (f64.neg (f64.load offset=16 (local.get $d))))
    (f64.store offset=24 (local.get $d) (f64.neg (f64.load offset=24 (local.get $d))))
    (f64.store offset=48 (local.get $d) (f64.neg (f64.load offset=48 (local.get $d))))
    (f64.store offset=56 (local.get $d) (f64.neg (f64.load offset=56 (local.get $d))))
    ;; Each of dX, dY and c rounded once from its exact sum, as $wedgeDepthAt's filter takes it.
    (call $byteCopy (i32.add (local.get $l) (i32.const 1584)) (local.get $d) (i32.const 32))
    (f64.store offset=64 (local.get $d)
      (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
        (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 4) (i32.add (local.get $l) (i32.const 4144)))))
    (call $byteCopy (i32.add (local.get $l) (i32.const 1584)) (i32.add (local.get $d) (i32.const 32)) (i32.const 32))
    (f64.store offset=72 (local.get $d)
      (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
        (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 4) (i32.add (local.get $l) (i32.const 4144)))))
    (call $byteCopy (i32.add (local.get $l) (i32.const 1584))
      (i32.add (i32.add (local.get $l) (i32.const 176)) (i32.mul (local.get $e) (i32.const 96))) (i32.const 32))
    (f64.store offset=80 (local.get $d)
      (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
        (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 4) (i32.add (local.get $l) (i32.const 4144)))))
    (local.set $a (i32.add (i32.add (local.get $l) (i32.const 112)) (i32.mul (local.get $e) (i32.const 96))))
    (local.set $b (i32.add (local.get $a) (i32.const 32)))
    (local.set $c (i32.add (local.get $a) (i32.const 64)))
    (local.set $o (f64.convert_i32_s (local.get $orientation)))
    ;; At or before the line: w <= W.
    (call $put (local.get $s) (i32.const 0) (i32.const 0) (local.get $a) (i32.const 4) (local.get $o))
    (call $put (local.get $s) (i32.const 1) (i32.const 0) (local.get $b) (i32.const 4) (local.get $o))
    (call $put (local.get $s) (i32.const 2) (i32.const 0) (local.get $c) (i32.const 4) (local.get $o))
    (local.set $s (call $closeSide (local.get $s)))
    ;; Inside the clip volume: w - x, w + x, w - y and w + y at least 0, which hold w at least 0 too.
    (block $volumeDone
      (loop $nextVolume
        (br_if $volumeDone (i32.ge_s (local.get $k) (i32.const 4)))
        (call $put (local.get $s) (i32.const 0) (i32.const 0) (local.get $a) (i32.const 4) (f64.neg (local.get $o)))
        (call $put (local.get $s) (i32.const 1) (i32.const 0) (local.get $b) (i32.const 4) (f64.neg (local.get $o)))
        (call $put (local.get $s) (i32.shr_u (local.get $k) (i32.const 1)) (i32.const 4) (local.get $c) (i32.const 4)
          (select (local.get $o) (f64.neg (local.get $o)) (i32.and (local.get $k) (i32.const 1))))
        (local.set $s (call $closeSide (local.get $s)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextVolume)))
    ;; At a depth from 0 to 1: z and w - z at least 0.
    (call $put (local.get $s) (i32.const 0) (i32.const 0) (local.get $d) (i32.const 4) (local.get $o))
    (call $put (local.get $s) (i32.const 1) (i32.const 0) (i32.add (local.get $d) (i32.const 32)) (i32.const 4) (local.get $o))
    (local.set $s (call $closeSide (local.get $s)))
    (call $put (local.get $s) (i32.const 0) (i32.const 0) (local.get $a) (i32.const 4) (f64.neg (local.get $o)))
    (call $put (local.get $s) (i32.const 0) (i32.const 4) (local.get $d) (i32.const 4) (f64.neg (local.get $o)))
    (call $put (local.get $s) (i32.const 1) (i32.const 0) (local.get $b) (i32.const 4) (f64.neg (local.get $o)))
    (call $put (local.get $s) (i32.const 1) (i32.const 4) (i32.add (local.get $d) (i32.const 32)) (i32.const 4) (f64.neg (local.get $o)))
    (local.set $s (call $closeSide (local.get $s)))
    ;; Within each edge of the triangle that runs through vec4(0): (xk - xj) Y - (yk - yj) X at least 0.
    (local.set $f (i32.const 0))
    (block $edgesDone
      (loop $nextEdge
        (br_if $edgesDone (i32.ge_s (local.get $f) (i32.const 3)))
        (if (i32.eqz (call $signOfTerms (i32.add (i32.add (local.get $l) (i32.const 176)) (i32.mul (local.get $f) (i32.const 96)))))
          (then
            (local.set $pj (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $f) (i32.const 1)) (i32.const 3)) (i32.const 5))))
            (local.set $pk (i32.add (local.get $l) (i32.shl (i32.rem_u (i32.add (local.get $f) (i32.const 2)) (i32.const 3)) (i32.const 5))))
            (call $put (local.get $s) (i32.const 0) (i32.const 0) (i32.add (local.get $pk) (i32.const 8)) (i32.const 1) (f64.neg (local.get $o)))
            (call $put (local.get $s) (i32.const 0) (i32.const 4) (i32.add (local.get $pj) (i32.const 8)) (i32.const 1) (local.get $o))
            (call $put (local.get $s) (i32.const 1) (i32.const 0) (local.get $pk) (i32.const 1) (local.get $o))
            (call $put (local.get $s) (i32.const 1) (i32.const 4) (local.get $pj) (i32.const 1) (f64.neg (local.get $o)))
            (local.set $s (call $closeSide (local.get $s)))))
        (local.set $f (i32.add (local.get $f) (i32.const 1)))
        (br $nextEdge)))
    (i32.store offset=9264 (local.get $l) (local.get $s))
    ;; Every fragment takes the weights of the triangle's point at vec4(0). Line f carries the scales of
    ;; the other two vertices, so its w coefficient over the scale of its own vertex weighs vertex f.
    (local.set $least (i32.const 0x7fffffff))
    (local.set $f (i32.const 0))
    (block $leastDone
      (loop $nextLeast
        (br_if $leastDone (i32.ge_s (local.get $f) (i32.const 3)))
        (local.set $exponent (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $f) (i32.const 2)))))
        (if (i32.and (f64.ne (f64.load offset=416 (i32.add (local.get $l) (i32.mul (local.get $f) (i32.const 24)))) (f64.const 0))
              (i32.lt_s (local.get $exponent) (local.get $least)))
          (then (local.set $least (local.get $exponent))))
        (local.set $f (i32.add (local.get $f) (i32.const 1)))
        (br $nextLeast)))
    (local.set $f (i32.const 0))
    (block $scaledDone
      (loop $nextScaled
        (br_if $scaledDone (i32.ge_s (local.get $f) (i32.const 3)))
        (local.set $weight (i32.add (i32.add (local.get $l) (i32.const 9360)) (i32.shl (local.get $f) (i32.const 3))))
        (f64.store (local.get $weight) (f64.load offset=416 (i32.add (local.get $l) (i32.mul (local.get $f) (i32.const 24)))))
        (if (f64.ne (f64.load (local.get $weight)) (f64.const 0))
          (then
            (local.set $exponent
              (i32.sub (local.get $least) (i32.load offset=96 (i32.add (local.get $l) (i32.shl (local.get $f) (i32.const 2))))))
            (call $multiplyByPowerOfTwo (local.get $weight)
              (select (local.get $exponent) (i32.const -1100) (i32.gt_s (local.get $exponent) (i32.const -1100))))))
        (local.set $f (i32.add (local.get $f) (i32.const 1)))
        (br $nextScaled)))
    (local.get $orientation))

  ;; The sign of side $s at the ray ($X, $Y, $W), summed exactly, as lines.ts's exactSideSign gives it.
  (func $exactSideSign (param $s i32) (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $k i32) (local $q i32) (local $n i32) (local $whole f64)
    (local.set $l (global.get $lines))
    (block $kDone
      (loop $nextK
        (br_if $kDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $whole (select (local.get $X) (select (local.get $Y) (local.get $W) (i32.eq (local.get $k) (i32.const 1))) (i32.eqz (local.get $k))))
        (local.set $q (i32.const 0))
        (block $qDone
          (loop $nextQ
            (br_if $qDone (i32.ge_s (local.get $q) (i32.const 8)))
            (call $product
              (f64.load (i32.add (i32.add (local.get $l) (i32.const 6824))
                (i32.shl (i32.add (i32.shl (i32.add (i32.mul (local.get $s) (i32.const 3)) (local.get $k)) (i32.const 3)) (local.get $q)) (i32.const 3))))
              (local.get $whole)
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $n) (i32.const 3))))
            (local.set $n (i32.add (local.get $n) (i32.const 2)))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $nextQ)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextK)))
    (call $signOf (i32.add (local.get $l) (i32.const 4144))
      (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (local.get $n) (i32.add (local.get $l) (i32.const 4144)))))

  ;; Whether the ray ($X, $Y, $W) lies inside every side of the triangle $prepareWedge prepared, or on
  ;; a side that owns it, as lines.ts's wedgeCovers decides it: each side's float value where that is
  ;; more than 2^-48 of its terms' sizes, its terms summed exactly otherwise.
  (func $wedgeCovers (param $X f64) (param $Y f64) (param $W f64) (result i32)
    (local $l i32) (local $s i32) (local $count i32) (local $c i32) (local $m i32) (local $value f64) (local $bound f64)
    (local $sign i32)
    (local.set $l (global.get $lines))
    (local.set $count (i32.load offset=9264 (local.get $l)))
    (block $done
      (loop $next
        (br_if $done (i32.ge_s (local.get $s) (local.get $count)))
        (local.set $c (i32.add (local.get $l) (i32.mul (local.get $s) (i32.const 24))))
        (local.set $value
          (f64.add
            (f64.add (f64.mul (f64.load offset=8744 (local.get $c)) (local.get $X)) (f64.mul (f64.load offset=8752 (local.get $c)) (local.get $Y)))
            (f64.mul (f64.load offset=8760 (local.get $c)) (local.get $W))))
        (local.set $bound
          (f64.mul (f64.const 0x1p-48)
            (f64.add
              (f64.add (f64.mul (f64.load offset=8984 (local.get $c)) (f64.abs (local.get $X))) (f64.mul (f64.load offset=8992 (local.get $c)) (f64.abs (local.get $Y))))
              (f64.mul (f64.load offset=9000 (local.get $c)) (local.get $W)))))
        (if (i32.eqz (f64.gt (local.get $value) (local.get $bound)))
          (then
            (if (f64.lt (local.get $value) (f64.neg (local.get $bound))) (then (return (i32.const 0))))
            (local.set $sign (call $exactSideSign (local.get $s) (local.get $X) (local.get $Y) (local.get $W)))
            (if (i32.or (i32.lt_s (local.get $sign) (i32.const 0))
                  (i32.and (i32.eqz (local.get $sign)) (i32.eqz (i32.load offset=9224 (i32.add (local.get $l) (i32.shl (local.get $s) (i32.const 2)))))))
              (then (return (i32.const 0))))))
        (local.set $s (i32.add (local.get $s) (i32.const 1)))
        (br $next)))
    (i32.const 1))

  ;; The depth at the ray ($X, $Y, $W) of the triangle $prepareWedge prepared, held to 0 to 1, as
  ;; lines.ts's wedgeDepthAt gives it: (dX X + dY Y) / (c W), summed exactly where its bound of
  ;; rounding, 2^-50 of its terms, passes 2^-32 of c W.
  (func $wedgeDepthAt (param $X f64) (param $Y f64) (param $W f64) (result f64)
    (local $l i32) (local $d i32) (local $x f64) (local $y f64) (local $divisor f64) (local $z f64) (local $q i32)
    (local.set $l (global.get $lines))
    (local.set $d (i32.add (local.get $l) (i32.const 9272)))
    (local.set $x (f64.mul (f64.load offset=64 (local.get $d)) (local.get $X)))
    (local.set $y (f64.mul (f64.load offset=72 (local.get $d)) (local.get $Y)))
    (local.set $divisor (f64.mul (f64.load offset=80 (local.get $d)) (local.get $W)))
    (local.set $z (f64.add (local.get $x) (local.get $y)))
    (if (f64.gt
          (f64.mul (f64.const 0x1p-50) (f64.add (f64.abs (local.get $x)) (f64.abs (local.get $y))))
          (f64.mul (f64.const 0x1p-32) (f64.abs (local.get $divisor))))
      (then
        (block $done
          (loop $next
            (br_if $done (i32.ge_s (local.get $q) (i32.const 4)))
            (call $product (f64.load (i32.add (local.get $d) (i32.shl (local.get $q) (i32.const 3)))) (local.get $X)
              (i32.add (i32.add (local.get $l) (i32.const 1584)) (i32.shl (local.get $q) (i32.const 4))))
            (call $product (f64.load offset=32 (i32.add (local.get $d) (i32.shl (local.get $q) (i32.const 3)))) (local.get $Y)
              (i32.add (i32.add (local.get $l) (i32.const 1648)) (i32.shl (local.get $q) (i32.const 4))))
            (local.set $q (i32.add (local.get $q) (i32.const 1)))
            (br $next)))
        (local.set $z
          (call $estimateOf (i32.add (local.get $l) (i32.const 4144))
            (call $addExactly (i32.add (local.get $l) (i32.const 1584)) (i32.const 16) (i32.add (local.get $l) (i32.const 4144)))))))
    (f64.min (f64.const 1) (f64.max (f64.const 0) (f64.div (local.get $z) (local.get $divisor)))))

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
  ;; into slot $to. A position whose four components all lie within 2^-48 of the largest sum the cut
  ;; takes, a component of $a and its step added by size, is rounding alone and is
  ;; vec4(0). One whose w alone is 0 keeps its x, y and z for the next plane to cut.
  (func $lerpToScratch
    (param $clipScratchBase i32) (param $varyingBytes i32) (param $varyingDescBase i32) (param $varyingDescCount i32)
    (param $a i32) (param $b i32) (param $t f64) (param $to i32)
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
    (param $src i32) (param $count i32) (param $dst i32) (param $plane i32) (result i32)
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
            (call $lerpToScratch (local.get $clipScratchBase) (local.get $varyingBytes) (local.get $varyingDescBase) (local.get $varyingDescCount)
              (local.get $a) (local.get $b)
              (f64.div (local.get $da) (f64.sub (local.get $da) (local.get $db)))
              (i32.add (local.get $dst) (local.get $out)))
            (local.set $out (i32.add (local.get $out) (i32.const 1)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $out))

  ;; ---- triangle pass ----
  ;; Interpolates each varying of the fragment from the varyings of three vertices at $v0, $v1
  ;; and $v2, each by its weight over $weights; a flat one is copied from $first, the varyings of
  ;; the triangle's first vertex.
  (func $interpolateVaryings
    (param $v0 i32) (param $v1 i32) (param $v2 i32) (param $first i32)
    (param $p0 f64) (param $p1 f64) (param $p2 f64) (param $weights f64)
    (local $d i32) (local $entry i32) (local $recordOffset i32) (local $fragmentDestAddress i32)
    (local $numComponents i32) (local $c i32) (local $offset i32)
    (local.set $d (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $d) (global.get $varyingDescCount)))
        (local.set $entry (call $descAddr (global.get $varyingDescBase) (local.get $d) (i32.const 20)))
        (local.set $recordOffset (i32.load (local.get $entry)))
        (local.set $fragmentDestAddress (i32.load (i32.add (local.get $entry) (i32.const 8))))
        (local.set $numComponents (i32.div_s (i32.load (i32.add (local.get $entry) (i32.const 12))) (i32.const 8)))
        ;; A flat varying is copied as the triangle's vertices all hold it, and none is interpolated.
        (if (i32.load (i32.add (local.get $entry) (i32.const 16)))
          (then
            (call $byteCopy
              (local.get $fragmentDestAddress)
              (i32.add (local.get $first) (local.get $recordOffset))
              (i32.load (i32.add (local.get $entry) (i32.const 12))))
            (local.set $numComponents (i32.const 0))))
        (local.set $c (i32.const 0))
        (block $innerBreak
          (loop $innerContinue
            (br_if $innerBreak (i32.ge_s (local.get $c) (local.get $numComponents)))
            (local.set $offset (i32.add (local.get $recordOffset) (i32.mul (local.get $c) (i32.const 8))))
            (f64.store
              (i32.add (local.get $fragmentDestAddress) (i32.mul (local.get $c) (i32.const 8)))
              (f64.div
                (f64.add
                  (f64.add
                    (f64.mul (local.get $p0) (f64.load (i32.add (local.get $v0) (local.get $offset))))
                    (f64.mul (local.get $p1) (f64.load (i32.add (local.get $v1) (local.get $offset)))))
                  (f64.mul (local.get $p2) (f64.load (i32.add (local.get $v2) (local.get $offset)))))
                (local.get $weights)))
            (local.set $c (i32.add (local.get $c) (i32.const 1)))
            (br $innerContinue)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $continue))))

  ;; Shades the fragment at pixel ($x, $y) of a triangle whose vertices' varyings lie at $v0, $v1
  ;; and $v2, and its first vertex's at $first, by the weights, their sum and the depth the lines
  ;; block holds at +6704: tests its depth, interpolates its varyings, runs the fragment stage and
  ;; writes what it gives.
  (func $shade (param $x i32) (param $y i32) (param $v0 i32) (param $v1 i32) (param $v2 i32) (param $first i32)
    (local $l i32) (local $pixelIndex i32) (local $depthAddr i32) (local $pixelDepth f64) (local $fragDepth f64)
    (local.set $l (global.get $lines))
    (local.set $pixelIndex (i32.add (i32.mul (local.get $y) (global.get $width)) (local.get $x)))
    (local.set $depthAddr (i32.add (global.get $depthBufferBase) (i32.mul (local.get $pixelIndex) (i32.const 8))))
    (local.set $pixelDepth (f64.load offset=6736 (local.get $l)))
    (block $skipPixel
      ;; LEQUAL depth test; a stage that writes its depth is tested once it has run.
      (br_if $skipPixel
        (i32.and (i32.eqz (global.get $writesDepth))
          (f64.gt (local.get $pixelDepth) (f64.load (local.get $depthAddr)))))
      (call $interpolateVaryings (local.get $v0) (local.get $v1) (local.get $v2) (local.get $first)
        (f64.load offset=6704 (local.get $l)) (f64.load offset=6712 (local.get $l)) (f64.load offset=6720 (local.get $l))
        (f64.load offset=6728 (local.get $l)))
      (if (global.get $readsFragCoord)
        (then
          (f64.store (global.get $fragCoordAddress) (f64.add (f64.convert_i32_s (local.get $x)) (f64.const 0.5)))
          (f64.store (i32.add (global.get $fragCoordAddress) (i32.const 8)) (f64.add (f64.convert_i32_s (local.get $y)) (f64.const 0.5)))))
      (if (global.get $mayDiscard) (then (i32.store (global.get $discardAddress) (i32.const 0))))
      ;; A stage that leaves its depth unwritten keeps the interpolated one.
      (if (global.get $usesDepth) (then (f64.store (global.get $fragDepthAddress) (local.get $pixelDepth))))
      (call $fragmentMain)
      ;; A fragment that discards leaves the pixel and its depth as they were.
      (if (global.get $mayDiscard) (then (br_if $skipPixel (i32.load (global.get $discardAddress)))))
      ;; A written depth is clamped to the depth range, as WebGPU clamps it.
      (local.set $fragDepth
        (select
          (f64.min (f64.max (f64.load (global.get $fragDepthAddress)) (f64.const 0)) (f64.const 1))
          (local.get $pixelDepth)
          (global.get $writesDepth)))
      (br_if $skipPixel
        (i32.and (global.get $writesDepth)
          (f64.gt (local.get $fragDepth) (f64.load (local.get $depthAddr)))))
      (f64.store (local.get $depthAddr) (local.get $fragDepth))
      ;; A fragment stage that writes no colour leaves the pixel as it is.
      (if (global.get $writesColour)
        (then
          (call $byteCopy
            (i32.add (global.get $outputBase) (i32.mul (local.get $pixelIndex) (i32.const 32)))
            (global.get $fragmentValueAddress)
            (i32.const 32))))))

  ;; Clips the triangle in scratch slots 0 to 2 to the clip volume, plane by plane, and gives how
  ;; many vertices the polygon left has; it starts at slot $clipped.
  (func $clip (result i32)
    (local $src i32) (local $count i32) (local $plane i32)
    (local.set $count (i32.const 3))
    (block $planeBreak
      (loop $planeContinue
        (br_if $planeBreak (i32.or (i32.ge_s (local.get $plane) (i32.const 6)) (i32.lt_s (local.get $count) (i32.const 3))))
        (local.set $count
          (call $clipPolygon (global.get $clipScratch) (global.get $varyingBytes) (global.get $varyingDescBase) (global.get $varyingDescCount)
            (local.get $src) (local.get $count) (i32.sub (i32.const 9) (local.get $src)) (local.get $plane)))
        (local.set $src (i32.sub (i32.const 9) (local.get $src)))
        (local.set $plane (i32.add (local.get $plane) (i32.const 1)))
        (br $planeContinue)))
    (global.set $clipped (local.get $src))
    (local.get $count))

  ;; Whether the position at $p is vec4(0), each component 0 or -0.
  (func $isOrigin (param $p i32) (result i32)
    (i32.and
      (i32.and (f64.eq (f64.load (local.get $p)) (f64.const 0)) (f64.eq (f64.load offset=8 (local.get $p)) (f64.const 0)))
      (i32.and (f64.eq (f64.load offset=16 (local.get $p)) (f64.const 0)) (f64.eq (f64.load offset=24 (local.get $p)) (f64.const 0)))))

  ;; Whether the position at $p lies in the clip volume: x and y from -w to w, z from 0 to w.
  (func $insideClipVolume (param $p i32) (result i32)
    (local $w f64)
    (local.set $w (f64.load offset=24 (local.get $p)))
    (i32.and
      (i32.and (f64.le (f64.abs (f64.load (local.get $p))) (local.get $w)) (f64.le (f64.abs (f64.load offset=8 (local.get $p))) (local.get $w)))
      (i32.and (f64.ge (f64.load offset=16 (local.get $p)) (f64.const 0)) (f64.le (f64.load offset=16 (local.get $p)) (local.get $w)))))

  ;; Rasterizes the triangle of vertices $i0, $i1 and $i2 by its own lines, which $prepareLines has
  ;; measured, as rasterizer.ts's rasterizeByLines does: the exact sign of each line decides
  ;; whether a centre lies inside, the point there must lie at a depth from 0 to 1, and each vertex weighs its line there, scaled; the vertex $centre takes all of it.
  (func $rasterizeByLines
    (param $positionsOutBase i32) (param $varyingsOutBase i32) (param $i0 i32) (param $i1 i32) (param $i2 i32)
    (param $centre i32) (param $orientation i32)
    (local $l i32) (local $widthF f64) (local $heightF f64) (local $k i32) (local $count i32) (local $i i32) (local $p i32)
    (local $left f64) (local $right f64) (local $top f64) (local $bottom f64) (local $sx f64) (local $sy f64) (local $bounded i32)
    (local $minX i32) (local $maxX i32) (local $minY i32) (local $maxY i32) (local $x i32) (local $y i32)
    (local $X f64) (local $Y f64) (local $W f64) (local $s0 i32) (local $s1 i32) (local $s2 i32) (local $mask i32)
    (local $l0 f64) (local $l1 f64) (local $l2 f64) (local $ratios i32) (local $inDepthRange i32)
    (local.set $l (global.get $lines))
    (local.set $widthF (f64.convert_i32_s (global.get $width)))
    (local.set $heightF (f64.convert_i32_s (global.get $height)))
    (local.set $minX (i32.const 0))
    (local.set $maxX (i32.sub (global.get $width) (i32.const 1)))
    (local.set $minY (i32.const 0))
    (local.set $maxY (i32.sub (global.get $height) (i32.const 1)))
    (local.set $left (f64.const inf))
    (local.set $right (f64.const -inf))
    (local.set $top (f64.const inf))
    (local.set $bottom (f64.const -inf))
    (if (i32.and
          (i32.and (f64.gt (f64.load offset=24 (local.get $l)) (f64.const 0)) (f64.gt (f64.load offset=56 (local.get $l)) (f64.const 0)))
          (f64.gt (f64.load offset=88 (local.get $l)) (f64.const 0)))
      (then
        (local.set $k (i32.const 0))
        (block $projected
          (loop $nextVertex
            (br_if $projected (i32.ge_s (local.get $k) (i32.const 3)))
            (local.set $p (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 5))))
            (local.set $sx (f64.mul (f64.add (f64.mul (f64.div (f64.load (local.get $p)) (f64.load offset=24 (local.get $p))) (f64.const 0.5)) (f64.const 0.5)) (local.get $widthF)))
            (local.set $sy (f64.mul (f64.sub (f64.const 1) (f64.add (f64.mul (f64.div (f64.load offset=8 (local.get $p)) (f64.load offset=24 (local.get $p))) (f64.const 0.5)) (f64.const 0.5))) (local.get $heightF)))
            (local.set $left (f64.min (local.get $left) (local.get $sx)))
            (local.set $right (f64.max (local.get $right) (local.get $sx)))
            (local.set $top (f64.min (local.get $top) (local.get $sy)))
            (local.set $bottom (f64.max (local.get $bottom) (local.get $sy)))
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br $nextVertex)))
        (local.set $minX (i32.trunc_sat_f64_s (f64.max (f64.floor (local.get $left)) (f64.const 0))))
        (local.set $maxX (i32.trunc_sat_f64_s (f64.min (f64.ceil (local.get $right)) (f64.sub (local.get $widthF) (f64.const 1)))))
        (local.set $minY (i32.trunc_sat_f64_s (f64.max (f64.floor (local.get $top)) (f64.const 0))))
        (local.set $maxY (i32.trunc_sat_f64_s (f64.min (f64.ceil (local.get $bottom)) (f64.sub (local.get $heightF) (f64.const 1))))))
      (else
        ;; The pixels to test are those of the polygon the clip leaves of the scaled vertices.
        (call $byteCopy (call $scratchVertexAddr (global.get $clipScratch) (global.get $varyingBytes) (i32.const 0)) (local.get $l) (i32.const 32))
        (call $byteCopy (call $scratchVertexAddr (global.get $clipScratch) (global.get $varyingBytes) (i32.const 1)) (i32.add (local.get $l) (i32.const 32)) (i32.const 32))
        (call $byteCopy (call $scratchVertexAddr (global.get $clipScratch) (global.get $varyingBytes) (i32.const 2)) (i32.add (local.get $l) (i32.const 64)) (i32.const 32))
        (local.set $count (call $clip))
        (if (i32.lt_s (local.get $count) (i32.const 3)) (then (return)))
        (local.set $bounded (i32.const 1))
        (block $polygonDone
          (loop $nextPolygonVertex
            (br_if $polygonDone (i32.ge_s (local.get $i) (local.get $count)))
            (local.set $p (call $scratchVertexAddr (global.get $clipScratch) (global.get $varyingBytes) (i32.add (global.get $clipped) (local.get $i))))
            (if (i32.eqz (f64.ge (f64.load offset=24 (local.get $p)) (f64.const 0x1p-24))) (then (local.set $bounded (i32.const 0))))
            (local.set $sx (f64.mul (f64.add (f64.mul (f64.div (f64.load (local.get $p)) (f64.load offset=24 (local.get $p))) (f64.const 0.5)) (f64.const 0.5)) (local.get $widthF)))
            (local.set $sy (f64.mul (f64.sub (f64.const 1) (f64.add (f64.mul (f64.div (f64.load offset=8 (local.get $p)) (f64.load offset=24 (local.get $p))) (f64.const 0.5)) (f64.const 0.5))) (local.get $heightF)))
            (local.set $left (f64.min (local.get $left) (local.get $sx)))
            (local.set $right (f64.max (local.get $right) (local.get $sx)))
            (local.set $top (f64.min (local.get $top) (local.get $sy)))
            (local.set $bottom (f64.max (local.get $bottom) (local.get $sy)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $nextPolygonVertex)))
        ;; The rounding of a cut moves a vertex by far less than half a pixel, which the floor and ceiling take in.
        (if (local.get $bounded)
          (then
            (local.set $minX (i32.trunc_sat_f64_s (f64.max (f64.floor (local.get $left)) (f64.const 0))))
            (local.set $maxX (i32.trunc_sat_f64_s (f64.min (f64.ceil (local.get $right)) (f64.sub (local.get $widthF) (f64.const 1)))))
            (local.set $minY (i32.trunc_sat_f64_s (f64.max (f64.floor (local.get $top)) (f64.const 0))))
            (local.set $maxY (i32.trunc_sat_f64_s (f64.min (f64.ceil (local.get $bottom)) (f64.sub (local.get $heightF) (f64.const 1)))))))))
    (local.set $W (f64.mul (local.get $widthF) (local.get $heightF)))
    ;; Every vertex at a depth from 0 to 1 in front of the eye, z from 0 to w and w above 0: a point
    ;; inside the lines is a sum of them with weights of one sign, so it needs no test of depth.
    (local.set $inDepthRange (i32.const 1))
    (local.set $k (i32.const 0))
    (block $rangeDone
      (loop $nextRange
        (br_if $rangeDone (i32.ge_s (local.get $k) (i32.const 3)))
        (local.set $p (i32.add (local.get $l) (i32.shl (local.get $k) (i32.const 5))))
        (if (i32.eqz
              (i32.and
                (i32.and (f64.gt (f64.load offset=24 (local.get $p)) (f64.const 0)) (f64.ge (f64.load offset=16 (local.get $p)) (f64.const 0)))
                (f64.le (f64.load offset=16 (local.get $p)) (f64.load offset=24 (local.get $p)))))
          (then (local.set $inDepthRange (i32.const 0))))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $nextRange)))
    (local.set $y (local.get $minY))
    (block $yBreak
      (loop $yContinue
        (br_if $yBreak (i32.gt_s (local.get $y) (local.get $maxY)))
        (local.set $Y (f64.convert_i32_s (i32.mul (i32.sub (i32.sub (global.get $height) (i32.shl (local.get $y) (i32.const 1))) (i32.const 1)) (global.get $width))))
        (call $prepareRow (local.get $Y) (local.get $W))
        (local.set $x (local.get $minX))
        (block $xBreak
          (loop $xContinue
            (br_if $xBreak (i32.gt_s (local.get $x) (local.get $maxX)))
            (local.set $X (f64.convert_i32_s (i32.mul (i32.sub (i32.add (i32.shl (local.get $x) (i32.const 1)) (i32.const 1)) (global.get $width)) (global.get $height))))
            (block $skip
              (local.set $s0 (i32.mul (call $lineSign (i32.const 0) (local.get $X) (local.get $Y) (local.get $W)) (local.get $orientation)))
              (br_if $skip (i32.or (i32.lt_s (local.get $s0) (i32.const 0)) (i32.and (i32.eqz (local.get $s0)) (i32.eqz (i32.load offset=544 (local.get $l))))))
              (local.set $s1 (i32.mul (call $lineSign (i32.const 1) (local.get $X) (local.get $Y) (local.get $W)) (local.get $orientation)))
              (br_if $skip (i32.or (i32.lt_s (local.get $s1) (i32.const 0)) (i32.and (i32.eqz (local.get $s1)) (i32.eqz (i32.load offset=548 (local.get $l))))))
              (local.set $s2 (i32.mul (call $lineSign (i32.const 2) (local.get $X) (local.get $Y) (local.get $W)) (local.get $orientation)))
              (br_if $skip (i32.or (i32.lt_s (local.get $s2) (i32.const 0)) (i32.and (i32.eqz (local.get $s2)) (i32.eqz (i32.load offset=552 (local.get $l))))))
              ;; At a depth from 0 to 1: z >= 0 and w - z >= 0. Inside the three lines, w is the
              ;; triangle's sign times W, so the point lies in front of the eye.
              (if (i32.eqz (local.get $inDepthRange))
                (then
                  (br_if $skip (i32.lt_s (i32.mul (call $pointSign (i32.const 2) (local.get $X) (local.get $Y) (local.get $W)) (local.get $orientation)) (i32.const 0)))
                  (br_if $skip (i32.lt_s (i32.mul (call $pointSign (i32.const -1) (local.get $X) (local.get $Y) (local.get $W)) (local.get $orientation)) (i32.const 0)))))
              (local.set $l0 (f64.load offset=752 (local.get $l)))
              (local.set $l1 (f64.load offset=760 (local.get $l)))
              (local.set $l2 (f64.load offset=768 (local.get $l)))
              ;; The depth there, held to 0 to 1, where coverage put it.
              (f64.store offset=6736 (local.get $l) (call $depthAt (local.get $X) (local.get $Y) (local.get $W)))
              (if (i32.ge_s (local.get $centre) (i32.const 0))
                (then
                  (f64.store offset=6704 (local.get $l) (f64.convert_i32_u (i32.eqz (local.get $centre))))
                  (f64.store offset=6712 (local.get $l) (f64.convert_i32_u (i32.eq (local.get $centre) (i32.const 1))))
                  (f64.store offset=6720 (local.get $l) (f64.convert_i32_u (i32.eq (local.get $centre) (i32.const 2)))))
                (else
                  ;; A line that is 0 there has a value of 0, so the mask only picks the ratios.
                  (local.set $mask
                    (i32.or (i32.or (i32.ne (local.get $s0) (i32.const 0)) (i32.shl (i32.ne (local.get $s1) (i32.const 0)) (i32.const 1)))
                      (i32.shl (i32.ne (local.get $s2) (i32.const 0)) (i32.const 2))))
                  (local.set $ratios (i32.add (i32.add (local.get $l) (i32.const 560)) (i32.mul (local.get $mask) (i32.const 24))))
                  (f64.store offset=6704 (local.get $l) (f64.mul (f64.load (local.get $ratios)) (f64.abs (local.get $l0))))
                  (f64.store offset=6712 (local.get $l) (f64.mul (f64.load offset=8 (local.get $ratios)) (f64.abs (local.get $l1))))
                  (f64.store offset=6720 (local.get $l) (f64.mul (f64.load offset=16 (local.get $ratios)) (f64.abs (local.get $l2))))))
              (f64.store offset=6728 (local.get $l)
                (f64.add (f64.add (f64.load offset=6704 (local.get $l)) (f64.load offset=6712 (local.get $l))) (f64.load offset=6720 (local.get $l))))
              (call $shade (local.get $x) (local.get $y)
                (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $i0) (global.get $varyingBytes)))
                (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $i1) (global.get $varyingBytes)))
                (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $i2) (global.get $varyingBytes)))
                (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $i0) (global.get $varyingBytes)))))
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (br $xContinue)))
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $yContinue))))

  ;; Draws the triangle of vertices $t0, $t1 and $t2, as rasterizer.ts's drawTriangle
  ;; does: one whose vertices do not lie in one plane with the eye, or with a vertex at vec4(0) and the other two in the clip volume, is
  ;; drawn by its lines; any other lies in one plane with the eye, and draws only where it contains vec4(0).
  (func $drawTriangle
    (param $positionsOutBase i32) (param $varyingsOutBase i32)
    (param $t0 i32) (param $t1 i32) (param $t2 i32)
    (local $a i32) (local $b i32) (local $c i32) (local $zeros i32) (local $centre i32) (local $orientation i32)
    (local.set $a (i32.add (local.get $positionsOutBase) (i32.mul (local.get $t0) (i32.const 32))))
    (local.set $b (i32.add (local.get $positionsOutBase) (i32.mul (local.get $t1) (i32.const 32))))
    (local.set $c (i32.add (local.get $positionsOutBase) (i32.mul (local.get $t2) (i32.const 32))))
    (local.set $zeros (i32.add (i32.add (call $isOrigin (local.get $a)) (call $isOrigin (local.get $b))) (call $isOrigin (local.get $c))))
    (local.set $centre (i32.const -1))
    (if (i32.eq (local.get $zeros) (i32.const 1))
      (then
        (local.set $centre (select (i32.const 0) (select (i32.const 1) (i32.const 2) (call $isOrigin (local.get $b))) (call $isOrigin (local.get $a))))
        (if (i32.eqz
              (i32.and
                (i32.and
                  (i32.or (i32.eqz (local.get $centre)) (call $insideClipVolume (local.get $a)))
                  (i32.or (i32.eq (local.get $centre) (i32.const 1)) (call $insideClipVolume (local.get $b))))
                (i32.or (i32.eq (local.get $centre) (i32.const 2)) (call $insideClipVolume (local.get $c)))))
          (then (local.set $centre (i32.const -1))))))
    (if (i32.or (i32.eqz (local.get $zeros)) (i32.ne (local.get $centre) (i32.const -1)))
      (then
        (local.set $orientation (call $prepareLines (local.get $a) (local.get $b) (local.get $c) (local.get $centre)))
        (if (local.get $orientation)
          (then
            ;; The sign of a triangle is its winding in normalized device coordinates, that of its part in front of the eye too.
            (if (f64.lt (f64.mul (f64.convert_i32_s (local.get $orientation)) (global.get $cullSign)) (f64.const 0))
              (then (return)))
            (call $rasterizeByLines (local.get $positionsOutBase) (local.get $varyingsOutBase)
              (local.get $t0) (local.get $t1) (local.get $t2)
              (local.get $centre) (local.get $orientation))
            (return)))
        (if (i32.ne (local.get $centre) (i32.const -1)) (then (return)))))
    (call $rasterizeWedge (local.get $varyingsOutBase) (local.get $a) (local.get $b) (local.get $c)
      (local.get $t0) (local.get $t1) (local.get $t2)))

  ;; Rasterizes the triangle whose positions are at $a, $b and $c, whose vertices lie in one plane with
  ;; the eye, as rasterizer.ts's rasterizeWedge does: where it contains vec4(0), the triangle from the
  ;; centre to the ends of the segment the rest of its part inside the clip volume projects to. Every
  ;; fragment takes the triangle's varyings at vec4(0), and the flat ones of its first vertex.
  (func $rasterizeWedge
    (param $varyingsOutBase i32) (param $a i32) (param $b i32) (param $c i32) (param $t0 i32) (param $t1 i32) (param $t2 i32)
    (local $l i32) (local $orientation i32) (local $x i32) (local $y i32) (local $X f64) (local $Y f64) (local $W f64)
    (local.set $l (global.get $lines))
    (local.set $orientation (call $prepareWedge (local.get $a) (local.get $b) (local.get $c)))
    (if (i32.eqz (local.get $orientation)) (then (return)))
    (if (f64.lt (f64.mul (f64.convert_i32_s (local.get $orientation)) (global.get $cullSign)) (f64.const 0)) (then (return)))
    (local.set $W (f64.convert_i32_s (i32.mul (global.get $width) (global.get $height))))
    (block $yBreak
      (loop $yContinue
        (br_if $yBreak (i32.ge_s (local.get $y) (global.get $height)))
        (local.set $Y (f64.convert_i32_s (i32.mul (i32.sub (i32.sub (global.get $height) (i32.shl (local.get $y) (i32.const 1))) (i32.const 1)) (global.get $width))))
        (local.set $x (i32.const 0))
        (block $xBreak
          (loop $xContinue
            (br_if $xBreak (i32.ge_s (local.get $x) (global.get $width)))
            (local.set $X (f64.convert_i32_s (i32.mul (i32.sub (i32.add (i32.shl (local.get $x) (i32.const 1)) (i32.const 1)) (global.get $width)) (global.get $height))))
            (if (call $wedgeCovers (local.get $X) (local.get $Y) (local.get $W))
              (then
                (f64.store offset=6704 (local.get $l) (f64.load offset=9360 (local.get $l)))
                (f64.store offset=6712 (local.get $l) (f64.load offset=9368 (local.get $l)))
                (f64.store offset=6720 (local.get $l) (f64.load offset=9376 (local.get $l)))
                (f64.store offset=6728 (local.get $l)
                  (f64.add (f64.add (f64.load offset=9360 (local.get $l)) (f64.load offset=9368 (local.get $l))) (f64.load offset=9376 (local.get $l))))
                (f64.store offset=6736 (local.get $l) (call $wedgeDepthAt (local.get $X) (local.get $Y) (local.get $W)))
                (call $shade (local.get $x) (local.get $y)
                  (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $t0) (global.get $varyingBytes)))
                  (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $t1) (global.get $varyingBytes)))
                  (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $t2) (global.get $varyingBytes)))
                  (i32.add (local.get $varyingsOutBase) (i32.mul (local.get $t0) (global.get $varyingBytes))))))
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (br $xContinue)))
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $yContinue))))

  ;; ---- rasterize: the vertex pass, then each triangle, drawn before the next ----
  (func $rasterize
    (param $vertexCount i32) (param $attrSrcBase i32) (param $attrStrideBytes i32)
    (param $attrDescBase i32) (param $attrDescCount i32) (param $vertexPositionAddress i32)
    (param $positionsOutBase i32) (param $width i32) (param $height i32)
    (param $fragmentValueAddress i32) (param $outputBase i32) (param $varyingBytes i32)
    (param $varyingDescBase i32) (param $varyingDescCount i32) (param $varyingsOutBase i32)
    (param $clipScratchBase i32)
    (param $depthBufferBase i32)
    ;; What the fragment stage writes or reads besides its colour: each address
    ;; is used only when the flag after it is 1.
    (param $writesColour i32)
    (param $fragCoordAddress i32) (param $readsFragCoord i32)
    (param $fragDepthAddress i32) (param $usesDepth i32) (param $writesDepth i32)
    (param $discardAddress i32) (param $mayDiscard i32)
    ;; The triangles to draw, three i32 vertex numbers each, wound as WebGPU winds
    ;; them, the first giving the flat varyings. A $triangleBase of 0 takes each
    ;; three vertices in turn instead.
    (param $triangleBase i32) (param $triangleCount i32)
    ;; The sign of the screen area of a triangle the draw culls, or 0 to cull none.
    (param $cullSign f64)
    (local $i i32) (local $tri i32) (local $t i32) (local $entry i32)
    (global.set $lines (call $linesBase (local.get $clipScratchBase) (local.get $varyingBytes)))
    (global.set $clipScratch (local.get $clipScratchBase))
    (global.set $varyingBytes (local.get $varyingBytes))
    (global.set $varyingDescBase (local.get $varyingDescBase))
    (global.set $varyingDescCount (local.get $varyingDescCount))
    (global.set $width (local.get $width))
    (global.set $height (local.get $height))
    (global.set $depthBufferBase (local.get $depthBufferBase))
    (global.set $outputBase (local.get $outputBase))
    (global.set $fragmentValueAddress (local.get $fragmentValueAddress))
    (global.set $writesColour (local.get $writesColour))
    (global.set $fragCoordAddress (local.get $fragCoordAddress))
    (global.set $readsFragCoord (local.get $readsFragCoord))
    (global.set $fragDepthAddress (local.get $fragDepthAddress))
    (global.set $usesDepth (local.get $usesDepth))
    (global.set $writesDepth (local.get $writesDepth))
    (global.set $discardAddress (local.get $discardAddress))
    (global.set $mayDiscard (local.get $mayDiscard))
    (global.set $cullSign (local.get $cullSign))

    ;; ---- vertex pass ----
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

    ;; ---- each triangle ----
    (block $trianglesDone
      (loop $nextTriangle
        (if (i32.eqz (local.get $triangleBase))
          (then
            (local.set $t (i32.mul (local.get $tri) (i32.const 3)))
            (br_if $trianglesDone (i32.ge_s (i32.add (local.get $t) (i32.const 2)) (local.get $vertexCount)))
            (call $drawTriangle (local.get $positionsOutBase) (local.get $varyingsOutBase)
              (local.get $t) (i32.add (local.get $t) (i32.const 1)) (i32.add (local.get $t) (i32.const 2))))
          (else
            (br_if $trianglesDone (i32.ge_s (local.get $tri) (local.get $triangleCount)))
            (local.set $entry (i32.add (local.get $triangleBase) (i32.mul (local.get $tri) (i32.const 12))))
            (call $drawTriangle (local.get $positionsOutBase) (local.get $varyingsOutBase)
              (i32.load (local.get $entry))
              (i32.load (i32.add (local.get $entry) (i32.const 4)))
              (i32.load (i32.add (local.get $entry) (i32.const 8))))))
        (local.set $tri (i32.add (local.get $tri) (i32.const 1)))
        (br $nextTriangle))))

  (export "rasterize" (func $rasterize)))
