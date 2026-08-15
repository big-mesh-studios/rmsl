type Listener = (...args: unknown[]) => void;

/**
 * A minimal event emitter, mirroring three.js's `EventDispatcher` shape:
 * `addEventListener`/`removeEventListener`/`dispatchEvent` and the
 * `dispatchEvent(event)` style where `event.type` names the event.
 */
export class EventDispatcher {
  private listeners: Map<string, Listener[]> = new Map();

  addEventListener(type: string, listener: Listener): void {
    const list = this.listeners.get(type);
    if (list) {
      if (!list.includes(listener)) list.push(listener);
    } else {
      this.listeners.set(type, [listener]);
    }
  }

  hasEventListener(type: string, listener: Listener): boolean {
    return this.listeners.get(type)?.includes(listener) ?? false;
  }

  removeEventListener(type: string, listener: Listener): void {
    const list = this.listeners.get(type);
    if (!list) return;
    const index = list.indexOf(listener);
    if (index !== -1) list.splice(index, 1);
  }

  dispatchEvent(event: { type: string } & Record<string, unknown>): void {
    const list = this.listeners.get(event.type);
    if (!list) return;
    for (const listener of [...list]) listener(event);
  }
}
