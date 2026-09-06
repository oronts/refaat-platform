# Failure-injection assurance

Refaat's safety properties are fail-closed: when a dependency it relies on fails, the runtime
refuses, denies, or degrades to a grounded fallback rather than serving something it cannot stand
behind. The assurance suite proves those properties hold under injected faults, so they are verified
and regression-locked, not just asserted in prose.

## How faults are injected

No runtime chaos capability ships; a runtime fault trigger would itself be an availability attack
surface. Faults enter only through the existing typed ports and store-layer test doubles, injected
into the real production adapters. Each test exercises the genuine fail-closed conversion (a store
outage becoming a deny, a provider outage becoming a degrade or refuse), and each has a control run
without the fault proving the same path succeeds, so no assertion is vacuous.

## The verified properties

| Injected fault | Property |
| --- | --- |
| Generation provider outage, grounding survives, safe-defaults policy | A grounded, cited degraded answer is served and marked degraded, and it is not cached. |
| Generation provider outage, grounding absent, safe-defaults policy | The query abstains with the honest no-answer, never an ungrounded answer. |
| Generation provider outage, no policy or refuse-all policy | The outage surfaces as a 503 refusal. |
| Kill-switch store outage, outside the stale window | The guarded operation is denied. |
| Kill-switch store outage, inside the operator stale window | Last-known-good is served for the bounded window before it too denies. |
| Quota store outage | Admission is denied. |
| Audit sink outage, fail-closed audit | The guarded operation refuses (503) rather than proceeding unaudited. |
| Audit sink outage, default best-effort audit | The operation proceeds and the failure is logged, proving the default posture is unchanged. |
| Egress denial | The destination is refused before any socket opens. |
| A status subsystem read failure | That subsystem degrades to unknown while the rest of the plane still reports and the snapshot never raises. |

## Honest framing

These are properties verified against injected faults and locked by a regression suite. They are not
a proof of impossibility: the suite is thorough but finite, and it covers the faults it injects. A
new dependency or a new outbound path must be added to the suite for its fail-closed property to be
covered.
