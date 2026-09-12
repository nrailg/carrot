# Distributed architecture

## Scope of the first milestone

The first milestone is a control-plane and data-plane skeleton. It deliberately
does not choose a model backend, an RL algorithm, or a robot environment.
Those integrations should depend on these primitives, not the reverse.

Ray is the required distributed runtime. Carrot hides Ray actors, object
references, queues, and scheduling strategies from training code rather than
trying to support an environment in which Ray is absent.

## Layers

### Resource and placement

`PlacementSpec` declares a named pool's bundle count, CPU/GPU capacity per
bundle, and Ray placement strategy. `Cluster.reserve()` creates that placement
group once. `RolePlacement` then selects logical bundle ranks and declares how
much capacity each actor consumes.

Multiple roles selecting the same pool and bundle ranks are colocated; roles
selecting disjoint ranks are isolated. Fractional actor resources let colocated
actors coexist in Ray's scheduler while the runner remains responsible for
model offload/onload and actual device-memory safety.

The private Ray layer probes each allocated bundle and sorts by node IP and
actual GPU ID before assigning logical bundle ranks. Carrot does not maintain a
second virtual model of Ray's nodes or devices.

### Runtime

The private Ray integration owns actors and channel lifecycles.
`ActorHandle` exposes only a worker rank and asynchronous method calls. It uses
Ray's `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` switches instead of rewriting
the caller's accelerator visibility environment, then maps Ray's assigned
physical device to `LOCAL_RANK` inside the actor.

### Worker and WorkerGroup

`Worker` contains process-local state and lifecycle hooks. Every worker reads
standard distributed environment variables directly:
`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT`.

`WorkerGroup` provides:

- broadcast with `call`;
- explicit scatter with `map` and `RankCall`;
- immutable rank selection with `select`;
- non-blocking `GroupResult`, with rank-aware errors.

Explicit scatter avoids overloading Python lists as both batches and per-rank
arguments. Immutable rank views avoid the one-shot mutable selection state used
by some WorkerGroup APIs.

### Channel

`Channel` is a bounded FIFO backed by `ray.util.queue.Queue`. The Ray queue is
hidden behind the public interface, so a handle can be passed directly to
worker calls without exposing Ray to training code.

Channels should carry domain schemas such as `TrajectoryBatch`; those schemas
belong in a future protocol package rather than in the runtime.

## Dependency direction

```text
runners (SFT / online RL / offline RL)
    ↓
domain protocols (batch / trajectory / weights)
    ↓
distributed API (cluster / group / channel)
    ↓
private Ray integration
```

Training and environment code must not import Ray. The private Ray integration
must not depend on algorithms, robot SDKs, or model engines.

## Reference-framework decisions

- Keep slime's short orchestration path, but do not expose Ray objects.
- Keep verl's resource/dispatch separation, but use explicit calls instead of
  decorated magic attributes in the initial API.
- Keep RLinf's placement, WorkerGroup, and Channel concepts, while hiding their
  Ray implementation.
- Preserve a route to G-Core's parallel-controller model: controller identity
  can later be added as worker environment without changing worker method APIs.
- Keep LeRobot as the likely persisted dataset format; do not make it the
  in-memory RPC protocol.

## Next milestones

1. Add typed `TensorBatch`, `TrajectoryBatch`, and weight-version protocols.
2. Add SFT and online-RL runner state machines against fake workers.
3. Add Ray placement groups with stable node/GPU ordering.
4. Add actor health checks and named actor reconnection.
5. Integrate one LeRobot dataset and one simulated environment end to end.
