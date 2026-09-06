# Governed fine-tuning

Refaat treats a fine-tuned model as a governed, durable, auditable artifact, not a
side effect. A fine-tune is requested, the training data is explicitly approved, a
durable worker job runs the configured provider, and the produced model reference
is registered as an immutable artifact. It is never promoted automatically: using
it requires the same explicit registry promotion every other artifact goes through.

Fine-tuning is off by default. A workspace that never requests a fine-tune gains
only one empty table. Nothing about fine-tuning enters the query answer cache or
any program artifact identity, so the runtime is byte-identical when it is unused.

## The governance flow

1. Request. `project.fine_tuning.request(dataset, base_model=..., requester=...)`
   creates a run in the `pending` state, not approved. The training data is a
   content-addressed `evaluation.Dataset`; its sha256 version is recorded as the
   provenance digest that ties the produced model to the exact data.
2. Approve. `project.fine_tuning.approve(run_id, approver)` gates the training
   data. A run can only be approved while it is `pending`, and only an approved run
   is allowed to execute. Four-eyes: approval requires a named approver and a run
   cannot be approved by its own requester, and over HTTP the approver is the
   server-resolved authenticated identity, never a client-supplied value. The job
   refuses to run an unapproved run. The identity
   compared here is the authenticated credential, so four-eyes is enforced at the
   credential level; binding it to a person across multiple credentials is the role of
   the identity directory in the enterprise governance layer.
3. Execute. `project.fine_tuning.enqueue(run_id)` enqueues a durable `fine_tune.run`
   job that the worker runs. A run reaches `succeeded` only when the provider
   confirms a terminal success with a trained model reference, never on job creation,
   and the produced model is registered as an immutable artifact but never promoted.
   A real fine-tune is long, so a not-yet-terminal job leaves the run `running` with
   its provider job reference persisted, and the worker re-enqueues a delayed poll of
   that same job (the interval is `REFAAT_FINE_TUNE_POLL_SECONDS`, default 60) so the
   run self-progresses to a terminal state without an operator re-running it. Polling
   reuses the persisted provider job rather than creating a second one, and the create
   call carries a content-addressed `Idempotency-Key` so even a crash between creating
   the provider job and persisting its reference cannot mint a duplicate job. Only the
   deterministic offline provider completes in one step. An operator-supplied provider
   `training_file` is trusted as an externally materialized training source; Refaat
   records it in provenance but does not verify its bytes against the approved dataset.
4. Promote. Using the produced model requires the operator's explicit
   `registry.promote(program, environment, artifact_id)`, the same approval-gated
   promotion path used for every other artifact.

Provenance is always recorded: the workspace, the dataset content-address digest,
the base model, the hyperparameters, the requester, the approver, and the
timestamps. A failed run stays inspectable with its error preserved in the
provenance, and no artifact is registered for a failed run.

## The honesty boundary: deterministic offline versus a real provider

Fine-tuning value needs a real training provider, so the machinery is built
deterministically and honestly:

- The default `DeterministicFineTuner` records the request and returns a
  content-addressed model reference (`ft:refaat:` plus the digest of the base
  model, the dataset version, and the hyperparameters). It performs no training
  and makes no network call. Its provenance says so plainly: it is a governance
  and provenance record, not a trained model. The whole gate runs on this provider
  with no paid model and no network, and it never fabricates a training result.
- The real `OpenAIFineTuningProvider` is selected only when
  `REFAAT_FINE_TUNING_PROVIDER=openai` is configured. It reads its key from config
  like the other model adapters and calls the provider's fine-tuning API. It is
  never exercised in the gate, so no network call ever enters a gate path.

A produced model reference is a proposal until it is promoted. The deterministic
reference is honest about not being a trained model, so it can drive the whole
governance flow in tests without claiming a capability it does not have.

## Durability

A run executes on the existing durable worker with the same lease, heartbeat, and
retry engine as every other job. A crash mid-run leaves the run resumable: the
job retries, and because the deterministic provider is idempotent and registration
is insert-if-absent, a resumed run converges. A run whose job exhausts its attempts
is terminalized to `failed` and stays inspectable.

## Surfaces

- Python: `project.fine_tuning.request(...)`, `.approve(...)`, `.enqueue(...)`,
  `.run(...)`, `.get(...)`.
- CLI: `refaat fine-tune request`, `approve`, `enqueue`, `run`, `status`.
- HTTP: `POST /v1/fine-tunes` (admin), `POST /v1/fine-tunes/{run_id}/approve`
  (admin), `POST /v1/fine-tunes/{run_id}/enqueue` (admin),
  `GET /v1/fine-tunes/{run_id}` (read).

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `REFAAT_FINE_TUNING_PROVIDER` | `deterministic` | `deterministic` (offline, no external resource) or `openai` (real training). |
| `REFAAT_FINE_TUNING_BASE_URL` | none | Base URL for the real provider; falls back to `REFAAT_OPENAI_BASE_URL`. |
| `REFAAT_FINE_TUNING_API_KEY` | none | Key for the real provider; falls back to `REFAAT_OPENAI_API_KEY`. |
| `REFAAT_FINE_TUNE_POLL_SECONDS` | `60` | Delay before a still-running provider job is re-polled by the worker. |

The deterministic default needs no external resource, so the offline gate never
needs a key.
