/** An arithmetic expression over the matrix's components `m[i]`, number literals and earlier locals. */
export type Formula =
  | { kind: "number"; value: number }
  | { kind: "input"; index: number }
  | { kind: "name"; name: string }
  | { kind: "negate"; operand: Formula }
  | { kind: "add" | "sub" | "mul" | "div"; left: Formula; right: Formula };

/** One statement of a matrix helper: locals it sets on one line, a component of its result, or the number it returns. */
export type FormulaStatement =
  | { target: "locals"; locals: [name: string, value: Formula][] }
  | { target: "out"; index: number; value: Formula }
  | { target: "return"; value: Formula };

/** The helpers the table holds: the inverse and the determinant of each square matrix. */
export type MatrixHelper = "mat2x2inv" | "mat3x3inv" | "mat4x4inv" | "mat2x2det" | "mat3x3det" | "mat4x4det";

const m = (index: number): Formula => ({ kind: "input", index });
const n = (name: string): Formula => ({ kind: "name", name });
const num = (value: number): Formula => ({ kind: "number", value });
const neg = (operand: Formula): Formula => ({ kind: "negate", operand });
const add = (left: Formula, right: Formula): Formula => ({ kind: "add", left, right });
const sub = (left: Formula, right: Formula): Formula => ({ kind: "sub", left, right });
const mul = (left: Formula, right: Formula): Formula => ({ kind: "mul", left, right });
const div = (left: Formula, right: Formula): Formula => ({ kind: "div", left, right });
const locals = (...entries: [string, Formula][]): FormulaStatement => ({ target: "locals", locals: entries });
const write = (index: number, value: Formula): FormulaStatement => ({ target: "out", index, value });
const returns = (value: Formula): FormulaStatement => ({ target: "return", value });

/**
 * The inverse and the determinant of each square matrix, as the statements
 * both CPU targets run, in order. The JS target prints them as the source of a
 * helper function, and WASM emits the same arithmetic, so the two agree to the
 * bit. An inverse writes every component of its result, and a determinant
 * returns a number.
 */
export const MATRIX_HELPER_FORMULAS: Record<MatrixHelper, readonly FormulaStatement[]> = {
  mat2x2inv: [
    locals(["a00", m(0)], ["a01", m(1)], ["a10", m(2)], ["a11", m(3)]),
    locals(["inv", div(num(1), sub(mul(n("a00"), n("a11")), mul(n("a01"), n("a10"))))]),
    write(0, mul(n("a11"), n("inv"))),
    write(1, mul(neg(n("a01")), n("inv"))),
    write(2, mul(neg(n("a10")), n("inv"))),
    write(3, mul(n("a00"), n("inv"))),
  ],
  mat3x3inv: [
    locals(["a00", m(0)], ["a01", m(1)], ["a02", m(2)]),
    locals(["a10", m(3)], ["a11", m(4)], ["a12", m(5)]),
    locals(["a20", m(6)], ["a21", m(7)], ["a22", m(8)]),
    locals(["b01", sub(mul(n("a22"), n("a11")), mul(n("a12"), n("a21")))]),
    locals(["b11", add(mul(neg(n("a22")), n("a10")), mul(n("a12"), n("a20")))]),
    locals(["b21", sub(mul(n("a21"), n("a10")), mul(n("a11"), n("a20")))]),
    locals(["det", add(add(mul(n("a00"), n("b01")), mul(n("a01"), n("b11"))), mul(n("a02"), n("b21")))]),
    locals(["inv", div(num(1), n("det"))]),
    write(0, mul(n("b01"), n("inv"))),
    write(1, mul(add(mul(neg(n("a22")), n("a01")), mul(n("a02"), n("a21"))), n("inv"))),
    write(2, mul(sub(mul(n("a12"), n("a01")), mul(n("a02"), n("a11"))), n("inv"))),
    write(3, mul(n("b11"), n("inv"))),
    write(4, mul(sub(mul(n("a22"), n("a00")), mul(n("a02"), n("a20"))), n("inv"))),
    write(5, mul(add(mul(neg(n("a12")), n("a00")), mul(n("a02"), n("a10"))), n("inv"))),
    write(6, mul(n("b21"), n("inv"))),
    write(7, mul(add(mul(neg(n("a21")), n("a00")), mul(n("a01"), n("a20"))), n("inv"))),
    write(8, mul(sub(mul(n("a11"), n("a00")), mul(n("a01"), n("a10"))), n("inv"))),
  ],
  mat4x4inv: [
    locals(["a00", m(0)], ["a01", m(1)], ["a02", m(2)], ["a03", m(3)]),
    locals(["a10", m(4)], ["a11", m(5)], ["a12", m(6)], ["a13", m(7)]),
    locals(["a20", m(8)], ["a21", m(9)], ["a22", m(10)], ["a23", m(11)]),
    locals(["a30", m(12)], ["a31", m(13)], ["a32", m(14)], ["a33", m(15)]),
    locals(["b00", sub(mul(n("a00"), n("a11")), mul(n("a01"), n("a10")))]),
    locals(["b01", sub(mul(n("a00"), n("a12")), mul(n("a02"), n("a10")))]),
    locals(["b02", sub(mul(n("a00"), n("a13")), mul(n("a03"), n("a10")))]),
    locals(["b03", sub(mul(n("a01"), n("a12")), mul(n("a02"), n("a11")))]),
    locals(["b04", sub(mul(n("a01"), n("a13")), mul(n("a03"), n("a11")))]),
    locals(["b05", sub(mul(n("a02"), n("a13")), mul(n("a03"), n("a12")))]),
    locals(["b06", sub(mul(n("a20"), n("a31")), mul(n("a21"), n("a30")))]),
    locals(["b07", sub(mul(n("a20"), n("a32")), mul(n("a22"), n("a30")))]),
    locals(["b08", sub(mul(n("a20"), n("a33")), mul(n("a23"), n("a30")))]),
    locals(["b09", sub(mul(n("a21"), n("a32")), mul(n("a22"), n("a31")))]),
    locals(["b10", sub(mul(n("a21"), n("a33")), mul(n("a23"), n("a31")))]),
    locals(["b11", sub(mul(n("a22"), n("a33")), mul(n("a23"), n("a32")))]),
    locals([
      "det",
      add(
        sub(
          add(
            add(sub(mul(n("b00"), n("b11")), mul(n("b01"), n("b10"))), mul(n("b02"), n("b09"))),
            mul(n("b03"), n("b08")),
          ),
          mul(n("b04"), n("b07")),
        ),
        mul(n("b05"), n("b06")),
      ),
    ]),
    locals(["inv", div(num(1), n("det"))]),
    write(0, mul(add(sub(mul(n("a11"), n("b11")), mul(n("a12"), n("b10"))), mul(n("a13"), n("b09"))), n("inv"))),
    write(1, mul(sub(add(mul(neg(n("a01")), n("b11")), mul(n("a02"), n("b10"))), mul(n("a03"), n("b09"))), n("inv"))),
    write(2, mul(add(sub(mul(n("a31"), n("b05")), mul(n("a32"), n("b04"))), mul(n("a33"), n("b03"))), n("inv"))),
    write(3, mul(sub(add(mul(neg(n("a21")), n("b05")), mul(n("a22"), n("b04"))), mul(n("a23"), n("b03"))), n("inv"))),
    write(4, mul(sub(add(mul(neg(n("a10")), n("b11")), mul(n("a12"), n("b08"))), mul(n("a13"), n("b07"))), n("inv"))),
    write(5, mul(add(sub(mul(n("a00"), n("b11")), mul(n("a02"), n("b08"))), mul(n("a03"), n("b07"))), n("inv"))),
    write(6, mul(sub(add(mul(neg(n("a30")), n("b05")), mul(n("a32"), n("b02"))), mul(n("a33"), n("b01"))), n("inv"))),
    write(7, mul(add(sub(mul(n("a20"), n("b05")), mul(n("a22"), n("b02"))), mul(n("a23"), n("b01"))), n("inv"))),
    write(8, mul(add(sub(mul(n("a10"), n("b10")), mul(n("a11"), n("b08"))), mul(n("a13"), n("b06"))), n("inv"))),
    write(9, mul(sub(add(mul(neg(n("a00")), n("b10")), mul(n("a01"), n("b08"))), mul(n("a03"), n("b06"))), n("inv"))),
    write(10, mul(add(sub(mul(n("a30"), n("b04")), mul(n("a31"), n("b02"))), mul(n("a33"), n("b00"))), n("inv"))),
    write(11, mul(sub(add(mul(neg(n("a20")), n("b04")), mul(n("a21"), n("b02"))), mul(n("a23"), n("b00"))), n("inv"))),
    write(12, mul(sub(add(mul(neg(n("a10")), n("b09")), mul(n("a11"), n("b07"))), mul(n("a12"), n("b06"))), n("inv"))),
    write(13, mul(add(sub(mul(n("a00"), n("b09")), mul(n("a01"), n("b07"))), mul(n("a02"), n("b06"))), n("inv"))),
    write(14, mul(sub(add(mul(neg(n("a30")), n("b03")), mul(n("a31"), n("b01"))), mul(n("a32"), n("b00"))), n("inv"))),
    write(15, mul(add(sub(mul(n("a20"), n("b03")), mul(n("a21"), n("b01"))), mul(n("a22"), n("b00"))), n("inv"))),
  ],
  mat2x2det: [returns(sub(mul(m(0), m(3)), mul(m(1), m(2))))],
  mat3x3det: [
    locals(["a00", m(0)], ["a01", m(1)], ["a02", m(2)]),
    locals(["a10", m(3)], ["a11", m(4)], ["a12", m(5)]),
    locals(["a20", m(6)], ["a21", m(7)], ["a22", m(8)]),
    returns(
      add(
        sub(
          mul(n("a00"), sub(mul(n("a11"), n("a22")), mul(n("a12"), n("a21")))),
          mul(n("a01"), sub(mul(n("a10"), n("a22")), mul(n("a12"), n("a20")))),
        ),
        mul(n("a02"), sub(mul(n("a10"), n("a21")), mul(n("a11"), n("a20")))),
      ),
    ),
  ],
  mat4x4det: [
    locals(["a00", m(0)], ["a01", m(1)], ["a02", m(2)], ["a03", m(3)]),
    locals(["a10", m(4)], ["a11", m(5)], ["a12", m(6)], ["a13", m(7)]),
    locals(["a20", m(8)], ["a21", m(9)], ["a22", m(10)], ["a23", m(11)]),
    locals(["a30", m(12)], ["a31", m(13)], ["a32", m(14)], ["a33", m(15)]),
    locals(["b00", sub(mul(n("a00"), n("a11")), mul(n("a01"), n("a10")))]),
    locals(["b01", sub(mul(n("a00"), n("a12")), mul(n("a02"), n("a10")))]),
    locals(["b02", sub(mul(n("a00"), n("a13")), mul(n("a03"), n("a10")))]),
    locals(["b03", sub(mul(n("a01"), n("a12")), mul(n("a02"), n("a11")))]),
    locals(["b04", sub(mul(n("a01"), n("a13")), mul(n("a03"), n("a11")))]),
    locals(["b05", sub(mul(n("a02"), n("a13")), mul(n("a03"), n("a12")))]),
    locals(["b06", sub(mul(n("a20"), n("a31")), mul(n("a21"), n("a30")))]),
    locals(["b07", sub(mul(n("a20"), n("a32")), mul(n("a22"), n("a30")))]),
    locals(["b08", sub(mul(n("a20"), n("a33")), mul(n("a23"), n("a30")))]),
    locals(["b09", sub(mul(n("a21"), n("a32")), mul(n("a22"), n("a31")))]),
    locals(["b10", sub(mul(n("a21"), n("a33")), mul(n("a23"), n("a31")))]),
    locals(["b11", sub(mul(n("a22"), n("a33")), mul(n("a23"), n("a32")))]),
    returns(
      add(
        sub(
          add(
            add(sub(mul(n("b00"), n("b11")), mul(n("b01"), n("b10"))), mul(n("b02"), n("b09"))),
            mul(n("b03"), n("b08")),
          ),
          mul(n("b04"), n("b07")),
        ),
        mul(n("b05"), n("b06")),
      ),
    ),
  ],
};

/** How tightly each operation binds in JS: a higher number binds tighter. */
const PRECEDENCE: Record<Formula["kind"], number> = {
  add: 1,
  sub: 1,
  mul: 2,
  div: 2,
  negate: 3,
  number: 4,
  input: 4,
  name: 4,
};

/** `e` as JS source, in parentheses wherever JS would otherwise group it differently. */
function formulaSource(e: Formula): string {
  const operand = (child: Formula, tighterThan: number) =>
    PRECEDENCE[child.kind] > tighterThan ? formulaSource(child) : `(${formulaSource(child)})`;
  switch (e.kind) {
    case "number":
      return String(e.value);
    case "input":
      return `m[${e.index}]`;
    case "name":
      return e.name;
    case "negate":
      return `-${operand(e.operand, PRECEDENCE.negate - 1)}`;
    default: {
      const symbol = { add: "+", sub: "-", mul: "*", div: "/" }[e.kind];
      const p = PRECEDENCE[e.kind];
      return `${operand(e.left, p - 1)} ${symbol} ${operand(e.right, p)}`;
    }
  }
}

/** The JS source of a matrix helper, `_mat3x3inv(m, out)` or `_mat3x3det(m)`, printed from the table. */
export function matrixHelperSource(helper: MatrixHelper): string {
  const statements = MATRIX_HELPER_FORMULAS[helper];
  const writes = statements.filter((s) => s.target === "out").length;
  const lines = statements.map((s) =>
    s.target === "locals"
      ? `let ${s.locals.map(([name, value]) => `${name} = ${formulaSource(value)}`).join(", ")};`
      : s.target === "out"
        ? `out[${s.index}] = ${formulaSource(s.value)};`
        : `return ${formulaSource(s.value)};`,
  );
  const body = writes > 0 ? [`out = out || new Array(${writes});`, ...lines, "return out;"] : lines;
  const params = writes > 0 ? "m, out" : "m";
  return `function _${helper}(${params}) {\n${body.map((line) => `  ${line}\n`).join("")}}`;
}
