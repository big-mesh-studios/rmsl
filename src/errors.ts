/** A backend rmsl compiles a program for and runs it on. */
export type Target = "glsl" | "wgsl" | "js" | "wasm";

/**
 * Where a call of rmsl runs: the backend, when the call has one, and the public
 * call the user made, which messages name.
 */
export interface Origin {
  target?: Target;
  caller: string;
}

/** The origin of a call of one backend, which every error of its own kinds names. */
export interface BackendOrigin extends Origin {
  target: Target;
}

/** What an error carries besides its code and its message. */
export interface RmslErrorOptions {
  /** The backend the error came from. */
  target?: Target;
  /** The error that caused this one. */
  cause?: unknown;
}

/**
 * The base class of every error rmsl throws. `code` names what went wrong more
 * finely than the class, and is the same on every target for one refusal.
 * `target` names the backend the error came from, where rmsl knows it.
 */
export class RmslError extends Error {
  override name = "RmslError";
  readonly code: string;
  declare target?: Target;

  constructor(code: string, message: string, options: RmslErrorOptions = {}) {
    super(message, "cause" in options ? { cause: options.cause } : undefined);
    this.code = code;
    if (options.target !== undefined) this.target = options.target;
  }
}

/** A mistake in the user's code that rmsl finds by reading the program, with no device. */
export class ProgramError extends RmslError {
  override name = "ProgramError";
}

/** A mistake in the user's code that shows only while the application runs. */
export class UsageError extends RmslError {
  override name = "UsageError";
}

/** A program that one target refuses and another runs. */
export class TargetError extends RmslError {
  override name = "TargetError";
  declare target: Target;

  constructor(code: string, message: string, options: RmslErrorOptions & { target: Target }) {
    super(code, message, options);
  }
}

/** A saved graph that `deserialize` cannot read. */
export class DataError extends RmslError {
  override name = "DataError";
  /** The index of the malformed entry of the saved graph, where one entry is at fault. */
  readonly index?: number;

  constructor(code: string, message: string, options: RmslErrorOptions & { index?: number } = {}) {
    super(code, message, options);
    if (options.index !== undefined) this.index = options.index;
  }
}

/** A platform the backend needs that is missing, such as WebGPU or a canvas context. */
export class PlatformError extends RmslError {
  override name = "PlatformError";
  declare target: Target;

  constructor(code: string, message: string, options: RmslErrorOptions & { target: Target }) {
    super(code, message, options);
  }
}

/** Work that does not fit the limits of the device. */
export class LimitError extends RmslError {
  override name = "LimitError";
  declare target: Target;
  /** The device's limit. */
  readonly limit: number;
  /** The amount the work asked for. */
  readonly requested: number;

  constructor(
    code: string,
    message: string,
    options: RmslErrorOptions & { target: Target; limit: number; requested: number },
  ) {
    super(code, message, options);
    this.limit = options.limit;
    this.requested = options.requested;
  }
}

/** A shader or another request the driver refused. */
export class DriverError extends RmslError {
  override name = "DriverError";
  declare target: Target;
  /** What the driver reported. */
  readonly log: string;

  constructor(code: string, message: string, options: RmslErrorOptions & { target: Target; log: string }) {
    super(code, message, options);
    this.log = options.log;
  }
}

/** A compiler of rmsl that found its own state inconsistent: a bug in rmsl. */
export class InternalError extends RmslError {
  override name = "InternalError";
}
