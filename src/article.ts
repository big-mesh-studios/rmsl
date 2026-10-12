/**
 * `word` after the article its sound takes, for a message that names a type,
 * a node or an array: "an int", "an Int32Array", "a uint", "a Uint32Array".
 */
export function withArticle(word: string): string {
  const vowel = /^[aeio]/i.test(word) || (/^u/i.test(word) && !/^u(int|vec|sampler|ni|se)/i.test(word));
  return `${vowel ? "an" : "a"} ${word}`;
}
