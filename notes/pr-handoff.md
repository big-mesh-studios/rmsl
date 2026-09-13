# Handing off: preparing the pull request

## State

`fix/compiler-codegen-bugs`, rebased onto current `origin/main` (base `7ca4e29`).
61 commits, 214 tests, 19 type tests, all green with real drivers. Typecheck clean.
`backup/pre-rebase` still points at the pre-rebase tip if anything needs comparing.

The rebase absorbed three upstream commits: version 1.0.6, `"files": ["dist"]`, and
Clinton's `dot()`/`length()` fix. His explicit result-type assignment was later
removed as redundant — the result-type table covers it — in `e1f8b3b`. Worth
telling him directly rather than leaving it to the diff.

## The draft

`/private/tmp/.../scratchpad/pr-draft.md` — copy it somewhere durable before the
scratchpad is cleaned up.

Structure: lead, a categorised overview table (New / Output changed / Now refused /
API changed, each with an "action needed" column), then New, Migration, Evidence,
Follow-ups.

It has been through a fact-check against real three.js source and two cold reads.
Those caught: five overstated claims, a migration example that contradicted its own
prose, a factually wrong statement about negative literals, and a `mod` recovery
line calling a method that does not exist.

## What is still wrong with it

**The description was written from session memory, not from the commit messages.**
That is its main weakness and the reason the `.node()` note was wrong: it explained
what `.node()` did, not why it existed. `739951d` and `959e353` say plainly that it
was a workaround for `Node<A>` being an intersection of fifteen conditionals, where
a self-referential member exhausted a 2 GB heap — and that removing it is only safe
because the registry encoding landed first.

The commit messages on this branch are careful and contain the reasoning the
description needs. Every migration heading should be checked against the commit
that made the change. Expect more cases like `.node()`.

Two errors of the same family were already found by checking against `origin/main`
rather than memory:

- A vertex result becoming the position is _pre-existing_; only the check is new.
- The implicit fragment colour output already existed on the WGSL side; only GLSL
  gained it, so that change is the two backends agreeing.

Neither cold reader can catch this class — they cannot see `origin/main` or the
history. It has to be done by reading commits.

## Suggested next steps

1. Walk `git log origin/main..HEAD` and check each migration heading in the draft
   against the commit that produced it. Correct anything that describes a change
   the branch did not make, or omits the reason it was made.
2. Decide the version. It is 1.0.6, matching main; the change set is breaking.
3. Decide whether the eight open findings in `code-review-2026-07-18.md` become a
   Follow-ups section, issues, or both.
4. Re-run one cold read on the corrected draft before publishing.

## Open findings

See `code-review-2026-07-18.md` in this directory: 44 findings, 36 fixed. The
remaining ones are quality rather than correctness — no known defect produces a
wrong or invalid shader.
