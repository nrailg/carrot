"""Run a tiny env-to-learner pipeline with local worker processes."""

from carrot.distributed import Cluster, PlacementSpec, RolePlacement, Worker


class EnvironmentWorker(Worker):
    def rollout(self, output, episode: int) -> None:
        output.put(
            {
                "episode": episode,
                "env_rank": self.rank,
                "reward": float(episode + self.rank),
            }
        )


class LearnerWorker(Worker):
    def update(self, trajectories: list[dict]) -> dict:
        return {
            "learner_rank": self.rank,
            "samples": len(trajectories),
            "mean_reward": sum(item["reward"] for item in trajectories) / len(trajectories),
        }


def main() -> None:
    with Cluster() as cluster:
        cluster.reserve("cpu", PlacementSpec(bundles_per_node=3))
        trajectories = cluster.channel("trajectories", maxsize=2)
        envs = cluster.launch(
            "env",
            EnvironmentWorker,
            placement=RolePlacement(pool="cpu", bundle_ranks=(0, 1)),
        )
        learner = cluster.launch(
            "learner",
            LearnerWorker,
            placement=RolePlacement(pool="cpu", bundle_ranks=(2,)),
        )

        envs.call("rollout", trajectories, 0).wait()
        batch = trajectories.get_batch(2, min_items=2)
        print(learner.call("update", batch).wait()[0])


if __name__ == "__main__":
    main()
