from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from .models import optional_import
from .types import Event


class ConsoleTracker:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream

    def log(self, event: Event) -> None:
        if event.kind in {"candidate", "finish"}:
            metrics = " ".join(f"{key}={value:.4f}" for key, value in event.metrics.items())
            print(f"[{event.kind} {event.step}] {metrics}", file=self.stream or sys.stderr)

    def close(self, status: str) -> None:
        pass


class JSONLTracker:
    """Append-only events; each fit is delimited by start and close records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, event: Event) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), allow_nan=False) + "\n")

    def close(self, status: str) -> None:
        self.log(Event("close", 0, {}, {"status": status}))


class MLflowTracker:
    """Owns an independent run; does not change MLflow's global active run."""

    def __init__(
        self, experiment: str = "prompt-optimiser", *, tracking_uri: str | None = None
    ) -> None:
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self._client: Any = None
        self._run_id: str | None = None

    def log(self, event: Event) -> None:
        if event.kind == "start":
            mlflow = optional_import("mlflow", "mlflow")
            self._client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            experiment = self._client.get_experiment_by_name(self.experiment)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self._client.create_experiment(self.experiment)
            )
            self._run_id = self._client.create_run(experiment_id).info.run_id
            for key, value in event.data.items():
                self._client.log_param(self._run_id, key, value)
        if self._run_id is None:
            return
        for key, value in event.metrics.items():
            self._client.log_metric(self._run_id, key, value, step=event.step)
        if event.kind == "artifact":
            self._client.log_artifacts(self._run_id, event.data["path"], artifact_path="experiment")
        if event.kind in {"candidate", "finish"}:
            name = f"candidate_{event.step}.json" if event.kind == "candidate" else "result.json"
            with tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / name
                artifact.write_text(json.dumps(event.data, indent=2, allow_nan=False))
                self._client.log_artifact(self._run_id, str(artifact))

    def close(self, status: str) -> None:
        if self._run_id is not None:
            run_id, self._run_id = self._run_id, None
            self._client.set_terminated(run_id, "FINISHED" if status == "finished" else "FAILED")


class WandbTracker:
    def __init__(self, project: str = "prompt-optimiser", **init_kwargs: Any) -> None:
        self.project = project
        self.init_kwargs = {"reinit": "create_new", **init_kwargs}
        self._run: Any = None

    def log(self, event: Event) -> None:
        if event.kind == "start":
            wandb = optional_import("wandb", "wandb")
            self._run = wandb.init(project=self.project, config=event.data, **self.init_kwargs)
        if self._run is None:
            return
        if event.metrics:
            self._run.log({**event.metrics, "candidate_step": event.step})
        if event.kind == "artifact":
            wandb = optional_import("wandb", "wandb")
            artifact = wandb.Artifact(f"prompt-{self._run.id}", type="prompt-experiment")
            artifact.add_dir(event.data["path"])
            self._run.log_artifact(artifact)
        if event.kind == "finish":
            self._run.summary.update(event.metrics)
            self._run.summary["best_prompt"] = event.data["prompt"]
            artifact = Path(self._run.dir) / "result.json"
            artifact.write_text(json.dumps(event.data, indent=2, allow_nan=False))
            self._run.save(str(artifact), base_path=self._run.dir, policy="now")

    def close(self, status: str) -> None:
        if self._run is not None:
            run, self._run = self._run, None
            run.finish(exit_code=0 if status == "finished" else 1)
