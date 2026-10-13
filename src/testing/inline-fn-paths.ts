import { Fn, float, If, instancedArray, int, Loop, uniform, vec3, While } from "../rmsl";

/** A program that reads an inline `Fn` on some paths, with the runs and the value it gives for each pair of conditions. */
export interface InlineFnPath {
  name: string;
  build: () => any;
  runs: (first: boolean, second: boolean) => number;
  value: (first: boolean, second: boolean) => number;
}

/**
 * Programs that read the value of an inline `Fn`, called outside their body,
 * on several paths. The `Fn` adds one to `counter` and returns 2, or a vector
 * whose first two components add to 3, and the program writes what it read to
 * `result`. `first` and `second` pick the branches.
 */
export function inlineFnPaths() {
  const counter = instancedArray(1, "float");
  const result = instancedArray(1, "float");
  const first = uniform("float");
  const second = uniform("float");
  const counted = () =>
    Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
  const out = () => result.element(int(0));
  const program = (body: (shared: any) => void) => () => {
    const shared = counted();
    return Fn(() => body(shared))();
  };
  const paths: InlineFnPath[] = [
    {
      name: "in a branch and after it",
      build: program((shared) => {
        If(first.greaterThan(0), () => {
          out().assign(shared);
        });
        out().addAssign(shared);
      }),
      runs: () => 1,
      value: (f) => (f ? 4 : 2),
    },
    {
      name: "in both branches of an If",
      build: program((shared) => {
        If(first.greaterThan(0), () => {
          out().assign(shared.add(1));
        }).Else(() => {
          out().assign(shared.mul(10));
        });
      }),
      runs: () => 1,
      value: (f) => (f ? 3 : 20),
    },
    {
      name: "twice inside one branch",
      build: program((shared) => {
        If(first.greaterThan(0), () => {
          out().assign(shared);
          out().addAssign(shared);
        });
      }),
      runs: (f) => (f ? 1 : 0),
      value: (f) => (f ? 4 : 0),
    },
    {
      name: "in both branches of an If nested in a branch",
      build: program((shared) => {
        If(first.greaterThan(0), () => {
          If(second.greaterThan(0), () => {
            out().assign(shared);
          }).Else(() => {
            out().assign(shared.mul(10));
          });
        });
      }),
      runs: (f) => (f ? 1 : 0),
      value: (f, s) => (f ? (s ? 2 : 20) : 0),
    },
    {
      name: "in a nested branch and after the outer one",
      build: program((shared) => {
        If(first.greaterThan(0), () => {
          If(second.greaterThan(0), () => {
            out().assign(shared);
          });
        });
        out().addAssign(shared);
      }),
      runs: () => 1,
      value: (f, s) => (f && s ? 4 : 2),
    },
    {
      name: "as a vector, twice in one statement",
      build: () => {
        const shared = Fn(() => {
          counter.element(int(0)).addAssign(1);
          return vec3(1, 2, 3);
        })() as any;
        return Fn(() => {
          out().assign(shared.x.add(shared.y));
        })();
      },
      runs: () => 1,
      value: () => 3,
    },
  ];
  return { counter, result, first, second, paths };
}

/** Every pair of conditions a path can take. */
export const CONDITION_PAIRS: [boolean, boolean][] = [
  [true, true],
  [true, false],
  [false, true],
  [false, false],
];

/** The loops whose condition reads an inline `Fn`: a `While` given a node or a function, and a `Loop(bool)`. */
export const CONDITION_LOOPS: [string, (test: () => any, body: () => void) => void][] = [
  ["a While condition given as a node", (test, body) => While(test(), body)],
  ["a While condition given as a function", (test, body) => While(() => test(), body)],
  ["a Loop condition", (test, body) => Loop(test(), body)],
];

/** Conditions that hold while `n` is below the value of `f`, reading that value once or twice. */
export const CONDITION_READS: [string, (n: any, f: any) => any][] = [
  ["read once", (n, f) => n.lessThan(f)],
  ["read twice", (n, f) => n.lessThan(f.add(f.mul(0)))],
];

/**
 * Programs whose only read of an inline `Fn`, called outside their body, is
 * the condition of a loop, each named by the loop and how often it reads. The
 * `Fn` adds one to `counter` and returns 3. The loop counts `n` up from 0 while
 * `n` is below that value, so the condition is tested four times, and the
 * program writes `n` to `result`.
 */
export function inlineFnConditions() {
  const counter = instancedArray(1, "float");
  const result = instancedArray(1, "float");
  const conditions = CONDITION_LOOPS.flatMap(([loop, write]) =>
    CONDITION_READS.map(([reads, test]) => ({
      name: `${loop}, ${reads}`,
      build: () => {
        const f = Fn(() => {
          counter.element(int(0)).addAssign(1);
          return float(3);
        })() as any;
        return Fn(() => {
          const n = float(0).toVar();
          write(
            () => test(n, f),
            () => n.addAssign(1),
          );
          result.element(int(0)).assign(n);
        })();
      },
    })),
  );
  return { counter, result, conditions };
}
