import { inflateSync } from "node:zlib";

/**
 * Reads the text of each page of a PDF that has no object streams and whose
 * fonts carry ToUnicode maps, as the GLSL ES 3.00 specification is written.
 * Lines break where the text moves to another baseline.
 */
export function pdfPages(pdf: Buffer): string[] {
  const raw = pdf.toString("latin1");
  const objects = new Map<number, { dict: string; stream?: Buffer }>();
  const objectPattern = /(\d+) 0 obj\b/g;
  let match: RegExpExecArray | null;
  while ((match = objectPattern.exec(raw))) {
    const start = match.index + match[0].length;
    const end = raw.indexOf("endobj", start);
    const body = raw.slice(start, end);
    const streamAt = body.search(/stream\r?\n/);
    if (streamAt < 0) {
      objects.set(Number(match[1]), { dict: body });
      continue;
    }
    const dataStart = start + streamAt + (body.slice(streamAt).startsWith("stream\r\n") ? 8 : 7);
    const dataEnd = raw.lastIndexOf("endstream", end);
    const dict = body.slice(0, streamAt);
    let data = pdf.subarray(dataStart, dataEnd);
    if (/FlateDecode/.test(dict)) data = inflateSync(data);
    objects.set(Number(match[1]), { dict, stream: data });
  }
  const object = (ref: string) => objects.get(Number(/(\d+) 0 R/.exec(ref)![1]))!;
  const resolveDict = (text: string, key: string): string | undefined => {
    const direct = new RegExp(`/${key}\\s*(\\d+ 0 R)`).exec(text);
    if (direct) return object(direct[1]!).dict;
    const inline = new RegExp(`/${key}\\s*<<`).exec(text);
    if (!inline) return undefined;
    let depth = 0;
    let i = inline.index + inline[0].length - 2;
    for (; i < text.length; i++) {
      if (text.startsWith("<<", i)) (depth++, i++);
      else if (text.startsWith(">>", i)) {
        depth--;
        i++;
        if (depth === 0) break;
      }
    }
    return text.slice(inline.index + inline[0].length - 2, i + 1);
  };

  const cmaps = new Map<string, { bytes: number; map: Map<number, string> }>();
  const cmapOf = (fontRef: string) => {
    const known = cmaps.get(fontRef);
    if (known) return known;
    const font = object(fontRef).dict;
    const toUnicode = /\/ToUnicode\s*(\d+ 0 R)/.exec(font);
    const map = new Map<number, string>();
    let bytes = 1;
    if (toUnicode) {
      const cmap = object(toUnicode[1]!).stream!.toString("latin1");
      const space = /begincodespacerange\s*<([0-9A-Fa-f]+)>/.exec(cmap);
      if (space) bytes = space[1]!.length / 2;
      const unicode = (hex: string) => String.fromCharCode(...(hex.match(/.{4}/g) ?? []).map((h) => parseInt(h, 16)));
      for (const block of cmap.match(/beginbfchar([\s\S]*?)endbfchar/g) ?? []) {
        for (const [, src, dst] of block.matchAll(/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>/g)) {
          map.set(parseInt(src!, 16), unicode(dst!));
        }
      }
      for (const block of cmap.match(/beginbfrange([\s\S]*?)endbfrange/g) ?? []) {
        for (const [, lo, hi, dst] of block.matchAll(/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>/g)) {
          const base = parseInt(dst!, 16);
          for (let code = parseInt(lo!, 16); code <= parseInt(hi!, 16); code++) {
            map.set(code, String.fromCharCode(base + code - parseInt(lo!, 16)));
          }
        }
      }
    }
    const entry = { bytes, map };
    cmaps.set(fontRef, entry);
    return entry;
  };

  const pages: string[] = [];
  const walk = (ref: string) => {
    const node = object(ref).dict;
    if (/\/Type\s*\/Pages\b/.test(node)) {
      const kids = /\/Kids\s*\[([^\]]*)\]/.exec(node)![1]!;
      for (const kid of kids.match(/\d+ 0 R/g) ?? []) walk(kid);
      return;
    }
    const fonts = new Map<string, string>();
    const resources = resolveDict(node, "Resources") ?? "";
    const fontDict = resolveDict(resources, "Font") ?? "";
    for (const [, name, fontRef] of fontDict.matchAll(/\/([\w.+-]+)\s*(\d+ 0 R)/g)) fonts.set(name!, fontRef!);
    const contents = /\/Contents\s*(\[[^\]]*\]|\d+ 0 R)/.exec(node)![1]!;
    const content = (contents.match(/\d+ 0 R/g) ?? []).map((r) => object(r).stream!.toString("latin1")).join("\n");
    let text = "";
    let cmap = { bytes: 1, map: new Map<number, string>() };
    let lastY: number | undefined;
    const decode = (hex: string) => {
      let out = "";
      for (let i = 0; i < hex.length; i += cmap.bytes * 2) {
        const code = parseInt(hex.slice(i, i + cmap.bytes * 2), 16);
        out += cmap.map.get(code) ?? "";
      }
      return out;
    };
    const tokens =
      /\/([\w.+-]+)\s+[\d.]+\s+Tf|([\d.-]+)\s+([\d.-]+)\s+T[dD]|<([0-9A-Fa-f]*)>\s*Tj|\[((?:<[0-9A-Fa-f]*>|[\s\d.-])*)\]\s*TJ/g;
    for (const token of content.matchAll(tokens)) {
      if (token[1] !== undefined) cmap = cmapOf(fonts.get(token[1])!);
      else if (token[2] !== undefined) {
        const y = Number(token[3]);
        if (lastY !== undefined && Math.abs(y - lastY) > 1) text += "\n";
        else text += " ";
        lastY = y;
      } else if (token[4] !== undefined) text += decode(token[4]);
      else {
        for (const [, hex, gap] of token[5]!.matchAll(/<([0-9A-Fa-f]*)>|(-?[\d.]+)/g)) {
          if (hex !== undefined) text += decode(hex);
          else if (Number(gap) < -200) text += " ";
        }
      }
    }
    pages.push(text);
  };
  const root = /\/Root\s*(\d+ 0 R)/.exec(raw)!;
  walk(/\/Pages\s*(\d+ 0 R)/.exec(object(root[1]!).dict)![1]!);
  return pages;
}
