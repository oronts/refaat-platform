# Prompt-injection posture

Retrieved evidence is untrusted data, not authority. An attacker who can place text
into a document that later gets retrieved will try to steer the model: to call a
forbidden tool, to fabricate a citation, or to override the instructions. Refaat's
posture is a bounded blast radius, not injection prevention. The structural
guarantees below hold regardless of what the retrieved text says, because none of
them consults document content to make a security decision.

## What Refaat does not claim

Refaat does not detect or prevent prompt injection semantically. There is no filter
that reads retrieved text and decides whether it is malicious, because a reliable
semantic filter needs a model and a model can be fooled. An attacker who controls
retrieved text can still influence the model's prose. What Refaat guarantees is that
such influence cannot cross a structural boundary: it cannot grant a tool, forge a
citation, or bypass access control.

## The held invariants

These four properties are structural and are locked by regression tests.

1. Tool permissions come from policy, not content. A tool is authorized only by the
   `ReActPolicy` grants and scopes the caller holds. `ToolGovernor.authorize` is a
   pure function of the policy and the tool name; it never reads a document, an
   observation, or the model's output. Injected text asking for a scope or a tool
   changes nothing, so an ungranted tool stays denied and its skill never runs.
2. Instruction and evidence channels are separated at the message-role level. The
   developer instructions occupy the system message; retrieved evidence occupies the
   user message. The two are never concatenated into one channel, so evidence text
   cannot rewrite the system instructions.
3. A citation can only be a retrieved, authorized chunk. Every cited id is filtered
   against the retrieved set before it becomes a citation, so a marker embedded in a
   document body that names another chunk, or a chunk the access policy dropped, is
   never surfaced as a citation. Under an active access policy the retrieved set is
   already the authorized set, so an ACL-dropped chunk cannot be cited even when a
   retrieved body forges its marker.
4. The offline path is immune to instruction-channel injection. The deterministic
   extractive generator ignores the instruction channel entirely: it never reads the
   system prompt, tunable instructions, or demonstrations, and it never parses
   citation markers out of text. Its answer and citations are a pure function of the
   retrieved evidence, so no injected instruction can change them.

## The untrusted-evidence envelope

When a real model is configured, each retrieved chunk body is rendered inside an
explicit untrusted-data envelope before it reaches the prompt:

```
<<UNTRUSTED_EVIDENCE chunk=... document=... score=...>>
[chunk:...]
chunk text
<<END_UNTRUSTED_EVIDENCE>>
```

Both online generators, the text generator and the vision generator, share one
helper (`refaat.core.evidence_framing.render_untrusted_evidence`) to render this, so
there is a single place that frames evidence for a model.

The envelope is legibility hardening, not a filter, and it is unfalsifiable: it
makes the trust boundary explicit to the model but makes no claim to neutralize an
injected instruction. Coverage is deliberately limited to retrieved chunk bodies.
Graph facts, the session summary, the bounded user profile, and supplemental harness
state remain plain labeled sections and are not wrapped. The offline extractive
generator does not use this helper, so its output is byte-identical with or without
the envelope, and the tested guarantees are the ones above, not the envelope.

## What is tested versus what is aspirational

The tested guarantees are invariants 1 through 4: the tool-authorization purity, the
role-level channel separation, the citation bound, and offline immunity. The
untrusted-evidence envelope is unfalsifiable legibility hardening layered on top of
those guarantees, not a property a test can prove. The promise is a bounded blast
radius, not prompt-injection prevention.
