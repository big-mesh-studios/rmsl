/**
 * `word` after the article its sound takes, for a message that names a type,
 * a node or an array: "an int", "an Int32Array", "a uint", "a Uint32Array",
 * "a URL", "an HTMLElement". A word that opens with two capitals is an
 * acronym, read letter by letter, so the name of its first letter decides.
 */
export function withArticle(word: string): string {
  const vowel = /^[A-Z]{2}/.test(word)
    ? /^[AEFHILMNORSX]/.test(word)
    : /^[aeio]/i.test(word) || (/^u/i.test(word) && !/^u(int|vec|sampler|norm|ni|se)/i.test(word));
  return `${vowel ? "an" : "a"} ${word}`;
}
