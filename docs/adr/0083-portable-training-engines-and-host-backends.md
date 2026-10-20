# ADR 0083: Separate portable training engines from host backend policy

Status: proposed
- Date: 2026-10-20
- Amends: [0017](0017-portable-training-backend-contract.md),
  [0020](0020-host-backend-policy.md), and
  [0082](0082-schema2-compose-envelope.md)

## Context

ADR 0017 keeps backend selection out of model composes so a compose stays
portable and machine policy stays on the machine. ADR 0082 adds a schema-2
`training.engine` and `architecture.provider` that select the experimental
Hugging Face Transformers implementation. Read literally, that is a backend
field in a compose, and 0082's accepted label does not resolve the conflict.

The two decisions describe different facts. MLX, PyTorch, and TorchTitan
execute one WALDO architecture and training definition; choosing between them
changes where and how the same model trains, not what model it is. A
Transformers `config_class`, its config values, and Trainer optimizer and
scheduler semantics define a different model and objective. They cannot be
derived from host policy, and two hosts that resolved different engines would
not be training the same compose.

## Decision

WALDO distinguishes three layers. Only the first may appear in a compose.

1. **Engine and provider (portable).** The implementation that gives the
   architecture and training parameters their meaning: `waldo-native` or
   `huggingface-transformers`, with the provider's pinned package, model and
   config classes, config, and allowlisted Trainer arguments. These are part of
   architecture and run identity. They do not name a host, device, framework
   build, path, or credential.
2. **Backend adapter (host policy).** The framework adapter that executes an
   engine on this machine, selected by `model.backend` under ADR 0020. The
   native engine resolves to MLX, PyTorch, TorchTitan, or explicit `fake`. The
   Transformers engine resolves only to its PyTorch-based Transformers worker.
   `auto` and `pytorch` select it; `mlx`, `torchtitan`, and `fake` conflict with
   the engine and fail before a run ID or run directory exists.
3. **Devices and topology (host policy).** Device kind and selection,
   visible accelerators, node count, world size, rank, and launch mechanism.
   These come from host configuration and environment, such as
   `WALDO_TRANSFORMERS_DEVICE` and `CUDA_VISIBLE_DEVICES`, never from a compose.
   Trainer arguments that select devices or distributed execution remain
   reserved and fail closed.

Portable execution semantics that happen to carry device-flavored provider
names, such as `per_device_train_batch_size`, `fp16`, `bf16`, and
`gradient_checkpointing`, are engine semantics, comparable to the native
precision and recomputation controls of ADR 0073. While the Transformers engine
is single-process, the per-device batch times gradient accumulation is the
global optimizer batch of ADR 0071. That resolved global batch, accumulation,
and precision are recorded as immutable run facts before training. A future
multi-process Transformers adapter must preserve that global batch rather than
scale it with the device count.

Schema-1 composes have no engine field and always mean `waldo-native`.
Schema-2 `waldo-native` composes normalize to the schema-1 domain
representation. Native architecture hashes, parent matching, resolved profiles,
checkpoint compatibility, backend selection, and run BOMs are unchanged.

ADR 0017's resolver contract applies to every engine: resolution returns the
adapter and immutable execution facts, including engine, provider package pin,
backend revision, framework, runtime, host, accelerators, node count, and world
size, before model or run state is created. A host that cannot execute the
compose's engine fails during resolution with the same reporting rules as a
missing framework.

Engine choice does not exempt a run from lifecycle contracts. ADR 0029,
0044, and 0079 apply to Transformers runs as written. Where the Transformers
engine does not yet meet them, the gap is unimplemented work, not an accepted
exception. Any engine-specific exception requires its own ADR.

## Consequences

- A compose remains portable across hosts that implement its engine. Selecting
  the Transformers engine narrows which hosts qualify; it never adds machine
  facts to the compose.
- `model.backend` keeps one meaning on every engine: a host preference among
  adapters that can execute the compose's engine.
- ADR 0082's envelope stands; its execution and device statements are read
  through this layering.
- Exact Transformers optimizer-state resume remains unimplemented and is not
  advertised. An interrupted Transformers stage restarts that stage.
- The maintained [compose](../MODEL-COMPOSE.md) and
  [lifecycle](../MODEL-LIFECYCLE.md) contracts must state the three layers and
  the Transformers backend preference rules when this ADR is accepted.
