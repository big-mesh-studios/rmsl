declare module "*.wat" {
  const bytes: Uint8Array;
  export default bytes;
  /** The same module importing its memory shared, or undefined when it imports none. */
  export const shared: Uint8Array | undefined;
}
