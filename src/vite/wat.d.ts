declare module "*.wat" {
  const bytes: Uint8Array;
  export default bytes;
  /** The same module importing its memory shared, when it imports one. */
  export const shared: Uint8Array | undefined;
}
