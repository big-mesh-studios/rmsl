import { UsageError, type Origin } from "../errors";
import { withArticle } from "../article";
import { componentCountOf } from "./cpu";

/**
 * Refuses `value`, given for the input `name` of `type`, unless it has the
 * shape of its type. A scalar takes one value, not an array, and a vector or
 * matrix an array of exactly its components, a matrix's column after column.
 * With `length`, the input is a uniform array, which takes an array of
 * exactly that many elements, each of that shape. A value left out, or
 * `null`, reads zero and passes, and so does any value of a sampler. A value
 * of the right shape costs one comparison and allocates nothing.
 */
export function assertHostShape(
  origin: Origin,
  what: string,
  name: string,
  type: string,
  value: unknown,
  length?: number,
): void {
  if (value === undefined || value === null || type.includes("sampler")) return;
  const count = componentCountOf(type);
  let wrong: any = value;
  let inEach = false;
  if (length === undefined) {
    if (count === 1 ? typeof value !== "object" : typeof value === "object" && (value as any).length === count) return;
  } else if (typeof value === "object" && (value as any).length === length) {
    for (let i = 0; i < length; i++) {
      const element = (value as any)[i];
      const fits =
        count === 1
          ? element === null || typeof element !== "object"
          : typeof element === "object" && element !== null && element.length === count;
      if (!fits) {
        wrong = element;
        inEach = true;
        break;
      }
    }
    if (!inEach) return;
  }
  const takes =
    length !== undefined && !inEach
      ? `${length} elements`
      : `${count === 1 ? "one value" : `${count} components`}${inEach ? " in each element" : ""}`;
  const given =
    typeof wrong === "object" && wrong !== null
      ? typeof wrong.length === "number"
        ? `an array of ${wrong.length}`
        : withArticle(wrong.constructor && wrong.constructor !== Object ? wrong.constructor.name : "object")
      : typeof wrong === "string"
        ? JSON.stringify(wrong)
        : String(wrong);
  throw new UsageError(
    "host-value-of-another-shape",
    `the ${type} ${what}${length === undefined ? "" : " array"} "${name}" takes ${takes}, not ${given}`,
    origin,
  );
}
