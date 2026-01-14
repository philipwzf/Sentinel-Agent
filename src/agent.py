import copy
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from pydantic import BaseModel, HttpUrl, ValidationError

from messenger import Messenger


logger = logging.getLogger("ai2thor_green")

REPO_ROOT = Path(__file__).resolve().parents[1]
SENTINEL_ROOT = REPO_ROOT / "SENTINEL_code"
EXAMPLES_PATH = REPO_ROOT / "examples" / "traj_data.json"
REWARD_CONFIG = SENTINEL_ROOT / "models" / "config" / "rewards.json"
SAFETY_RULES = SENTINEL_ROOT / "safety_rules_object.json"
MODEL_NAME = "ai2thor-agent"
MODEL_SPLIT = "examples"
MAX_STEPS = 200
MAX_FAILS = 5
PURPLE_ROLE = "agent"


def ensure_sentinel_on_path() -> None:
    if SENTINEL_ROOT.exists():
        sentinel_root_str = str(SENTINEL_ROOT)
        if sentinel_root_str not in sys.path:
            sys.path.insert(0, sentinel_root_str)


ensure_sentinel_on_path()
from models.eval.eval_llm import EvalLLM


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""

    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class EvalConfig(BaseModel):
    """Config schema for this green agent."""

    num_trials: int


@dataclass
class TrialSpec:
    trial_id: str
    traj_data: dict[str, Any]
    goal_instruction: str
    metadata: dict[str, Any]


class EpisodeTrace:
    """Collect per-step execution records for a single episode."""

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._step_index = 0

    def record(self, plan_action, thor_action, success, error, event_metadata) -> None:
        entry = {
            "step": self._step_index,
            "plan_action": self._sanitize(plan_action),
            "thor_action": self._sanitize(thor_action),
            "success": bool(success),
            "error": error or "",
            "event_metadata": self._sanitize(event_metadata) if event_metadata is not None else None,
        }
        self._steps.append(entry)
        self._step_index += 1

    def export(self) -> list[dict[str, Any]]:
        return list(self._steps)

    @staticmethod
    def _sanitize(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): EpisodeTrace._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [EpisodeTrace._sanitize(v) for v in value]
        if hasattr(value, "tolist"):
            return EpisodeTrace._sanitize(value.tolist())
        return str(value)


class ActionRunner:
    """Execute action lists in AI2-THOR while capturing traces."""

    def __init__(self, env) -> None:
        self.env = env
        self._trace: EpisodeTrace | None = None

    def run(self, actions: list[dict[str, Any]]) -> tuple[EpisodeTrace, int, int]:
        self._trace = EpisodeTrace()
        failures = 0
        step_idx = 0

        for action in actions:
            if step_idx >= MAX_STEPS:
                break

            if not isinstance(action, dict):
                self._record_step(action, None, False, "Action must be a dict", None)
                break

            action_name = action.get("action")
            if not action_name:
                self._record_step(action, None, False, "Missing action", None)
                break

            if action_name.lower() in {"stop", "end", "finish", "done"}:
                self._exec_thor_action({"action": "Done"})
                break

            success, _event, _error = self.execute_action(action)
            if not success:
                failures += 1
                if failures >= MAX_FAILS:
                    break

            step_idx += 1

        trace = self._trace
        self._trace = None
        return trace or EpisodeTrace(), step_idx, failures

    def execute_action(self, action_dict: dict[str, Any]) -> tuple[bool, Any, str]:
        action_name = action_dict.get("action")
        if action_name == "GotoLocation":
            try:
                return self._execute_goto(action_dict)
            except Exception as exc:
                self._record_step(action_dict, None, False, str(exc), None)
                return False, None, str(exc)
        return self._exec_thor_action(action_dict)

    def _exec_thor_action(self, action_dict: dict[str, Any]) -> tuple[bool, Any, str]:
        action_name = action_dict.get("action")
        object_id = action_dict.get("object_id") or action_dict.get("objectId", "")
        receptacle_id = action_dict.get("receptacle_id") or action_dict.get(
            "receptacleObjectId", ""
        )
        agent_id = action_dict.get("agentId")

        if action_name == "PutObject":
            if not receptacle_id:
                self._record_step(action_dict, None, False, "PutObject missing receptacle_id", None)
                return False, None, "PutObject missing receptacle_id"
            object_id = receptacle_id

        try:
            event, api_action = self.env.to_thor_api_exec(
                action_name,
                object_id,
                smooth_nav=False,
                agent_id=agent_id,
            )
            success = event.metadata["lastActionSuccess"]
            error = event.metadata.get("errorMessage", "") if not success else ""
            self._record_step(action_dict, api_action, success, error, event)
            return success, event, error
        except Exception as exc:
            self._record_step(action_dict, None, False, str(exc), None)
            return False, None, str(exc)

    def _execute_goto(self, action_dict: dict[str, Any]) -> tuple[bool, Any, str]:
        ensure_sentinel_on_path()
        from nav_astar import (
            astar_actions,
            build_adjacency,
            build_node_lookup,
            jps_actions,
            nearest_key,
            normalize_yaw as snap_yaw,
        )

        target_object_id = action_dict.get("object_id") or action_dict.get("objectId")
        if not target_object_id:
            return False, self.env.last_event, "GotoLocation missing target object_id"

        reachable = self.env.step(action="GetReachablePositions")
        reach_meta = reachable.metadata
        if not reach_meta:
            return False, self.env.last_event, "No reachable positions found"
        agent_meta = reach_meta["agent"]
        reachable_positions = reach_meta["actionReturn"]

        event = self.env.step(
            action="GetInteractablePoses",
            objectId=target_object_id,
            positions=reachable_positions,
            horizons=[float(h) for h in range(-30, 61, 3)],
            standings=[True],
        )
        poses = event.metadata["actionReturn"]
        if not poses:
            raise RuntimeError("No interactable poses returned for target object")

        target_position = poses[0]
        if target_position is None:
            return False, self.env.last_event, "GotoLocation missing valid target"

        nodes = build_node_lookup(reachable_positions)
        agent_position = agent_meta["position"]

        start_key = nearest_key(nodes, agent_position)
        target_key = nearest_key(nodes, target_position)
        start_yaw = snap_yaw(agent_meta["rotation"]["y"])

        try:
            action_names = jps_actions(nodes, start_key, start_yaw, target_key)
        except RuntimeError:
            adjacency = build_adjacency(nodes)
            action_names = astar_actions(nodes, adjacency, start_key, start_yaw, target_key)

        for action_name in action_names:
            success, event, error = self._exec_thor_action({"action": action_name})
            if not success:
                return success, event, error

        event = self.env.last_event
        agent_meta = event.metadata["agent"] if event else {}
        current_yaw = agent_meta.get("rotation", {}).get("y")

        desired_yaw = target_position.get("rotation")
        for turn_name in self._turn_actions(current_yaw, desired_yaw):
            success, event, error = self._exec_thor_action({"action": turn_name})
            if not success:
                return success, event, error

        if event and target_object_id:
            visible = self._is_object_visible(event.metadata, target_object_id)
            if not visible:
                visible = self._adjust_horizon_for_visibility(target_object_id)
                event = self.env.last_event
            success = visible
        else:
            success = False

        error = "" if success else (event.metadata.get("errorMessage", "") if event else "")
        return success, event, error

    @staticmethod
    def _turn_actions(current_yaw: float | None, desired_yaw: float | None) -> list[str]:
        ensure_sentinel_on_path()
        from nav_astar import normalize_yaw as snap_yaw

        if desired_yaw is None or current_yaw is None:
            return []
        current = snap_yaw(current_yaw)
        desired = snap_yaw(desired_yaw)
        diff = (desired - current) % 360
        if diff == 0:
            return []
        if diff == 90:
            return ["RotateRight"]
        if diff == 180:
            return ["RotateRight", "RotateRight"]
        if diff == 270:
            return ["RotateLeft"]
        if diff < 180:
            steps = int(round(diff / 90.0))
            return ["RotateRight"] * steps
        steps = int(round((360.0 - diff) / 90.0))
        return ["RotateLeft"] * steps

    def _adjust_horizon_for_visibility(self, target_object_id: str) -> bool:
        event = self.env.last_event
        if event is None or not getattr(event, "metadata", None):
            return False

        min_horizon = -30.0
        max_horizon = 90.0

        latest_event = event
        agent_meta = latest_event.metadata.get("agent", {}) if latest_event.metadata else {}
        current_horizon = agent_meta.get("cameraHorizon", 0.0)

        if self._is_object_visible(latest_event.metadata, target_object_id):
            return True

        while current_horizon > min_horizon + 1e-3:
            lookup_success, latest_event, _ = self._exec_thor_action({"action": "LookUp"})
            if not lookup_success:
                raise RuntimeError("LookUp action failed during horizon adjustment")
            if latest_event and self._is_object_visible(latest_event.metadata, target_object_id):
                return True
            agent_meta = latest_event.metadata.get("agent", {}) if latest_event.metadata else {}
            current_horizon = agent_meta.get("cameraHorizon", current_horizon)

        while current_horizon < max_horizon - 1e-3:
            lookdown_success, latest_event, _ = self._exec_thor_action({"action": "LookDown"})
            if not lookdown_success:
                raise RuntimeError("LookDown action failed during horizon adjustment")
            if latest_event and self._is_object_visible(latest_event.metadata, target_object_id):
                return True
            agent_meta = latest_event.metadata.get("agent", {}) if latest_event.metadata else {}
            current_horizon = agent_meta.get("cameraHorizon", current_horizon)

        raise RuntimeError("Horizon adjustment exceeded limits without finding target object")

    @staticmethod
    def _is_object_visible(metadata: dict[str, Any] | None, target_object_id: str) -> bool:
        if not metadata:
            return False
        objects = metadata.get("objects", [])
        for obj in objects:
            if obj.get("objectId") == target_object_id:
                return bool(obj.get("visible", False))
        return False

    def _record_step(self, plan_action, thor_action, success, error, event) -> None:
        if self._trace is None:
            return
        metadata = event.metadata if event is not None else None
        cleaned = EvalLLM.remove_useless_info(metadata) if isinstance(metadata, dict) else None
        self._trace.record(plan_action, thor_action, success, error, cleaned)


class Agent:
    required_roles: list[str] = [PURPLE_ROLE]
    required_config_keys: list[str] = ["num_trials"]

    def __init__(self) -> None:
        self.messenger = Messenger()
        ensure_sentinel_on_path()
        if not SENTINEL_ROOT.exists():
            raise FileNotFoundError(f"Missing SENTINEL_code directory at {SENTINEL_ROOT}")

        from env.thor_env import ThorEnv

        self.ThorEnv = ThorEnv

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        if request.config.get("num_trials", 0) <= 0:
            return False, "num_trials must be > 0"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
            config = EvalConfig.model_validate(request.config)
        except ValidationError as exc:
            await updater.reject(new_agent_text_message(f"Invalid request: {exc}"))
            return

        purple_url = request.participants.get(PURPLE_ROLE)
        if not purple_url:
            await updater.reject(new_agent_text_message(f"Missing participant role: {PURPLE_ROLE}"))
            return

        if not EXAMPLES_PATH.exists():
            await updater.reject(new_agent_text_message(f"traj_path not found: {EXAMPLES_PATH}"))
            return
        if not REWARD_CONFIG.exists():
            await updater.reject(new_agent_text_message(f"reward_config not found: {REWARD_CONFIG}"))
            return

        try:
            trials = load_trials(EXAMPLES_PATH, config.num_trials)
        except Exception as exc:
            await updater.reject(new_agent_text_message(f"Failed to load trials: {exc}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Preparing scene metadata for purple agent."),
        )

        trial_specs: list[TrialSpec] = []
        env = self.ThorEnv(headless=True)
        try:
            for idx, trial_data in enumerate(trials, start=1):
                trial_id = str(trial_data.get("task_id") or f"trial_{idx}")
                goal_instruction = trial_data.get("task_desc", "")
                scene_metadata = setup_scene(env, trial_data)
                trial_specs.append(
                    TrialSpec(
                        trial_id=trial_id,
                        traj_data=trial_data,
                        goal_instruction=goal_instruction,
                        metadata=scene_metadata,
                    )
                )
        finally:
            env.stop()

        purple_payload = {
            "instructions": (
                "Return a JSON dict mapping trial_id to a list of action dicts. "
                "Each action dict should include 'action' and optional 'object_id', "
                "'receptacle_id', or 'agentId'."
            ),
            "trials": [
                {
                    "trial_id": spec.trial_id,
                    "goal_instruction": spec.goal_instruction,
                    "metadata": spec.metadata,
                }
                for spec in trial_specs
            ],
        }

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Requesting action lists from purple agent."),
        )

        response_text = await self.messenger.talk_to_agent(
            message=json.dumps(purple_payload),
            url=str(purple_url),
            new_conversation=True,
        )

        try:
            response_payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            await updater.failed(new_agent_text_message(f"Purple response is not JSON: {exc}"))
            return

        if isinstance(response_payload, dict) and "actions" in response_payload:
            action_map = response_payload["actions"]
        else:
            action_map = response_payload
        if not isinstance(action_map, dict):
            await updater.failed(new_agent_text_message("Purple response must be a dict of action lists"))
            return

        logs_root = SENTINEL_ROOT / "logs" / "trajectories" / MODEL_NAME / MODEL_SPLIT
        logs_root.mkdir(parents=True, exist_ok=True)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Executing action lists in AI2-THOR."),
        )

        results: list[dict[str, Any]] = []
        env = self.ThorEnv()
        runner = ActionRunner(env)

        try:
            for spec in trial_specs:
                actions = action_map.get(spec.trial_id)
                if not isinstance(actions, list):
                    results.append(
                        {
                            "trial_id": spec.trial_id,
                            "error": "No action list returned for trial",
                        }
                    )
                    continue

                setup_scene(env, spec.traj_data)
                trace, steps, failures = runner.run(actions)

                success = env.get_goal_satisfied()
                completed, total = env.get_goal_conditions_met()
                goal_condition_success_rate = (completed / float(total)) if total else 0.0

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                trial_dir = logs_root / spec.trial_id
                trial_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trial_dir / f"r0_{timestamp}.json"
                log_path = trial_dir / f"r0_{timestamp}.txt"

                trace_payload = {
                    "trajectory": trace.export(),
                    "success": bool(success),
                }
                with trace_path.open("w", encoding="utf-8") as handle:
                    json.dump(trace_payload, handle, indent=2)

                log_lines = [
                    f"trial_id={spec.trial_id}",
                    f"success={success}",
                    f"completed_goal_conditions={completed}",
                    f"total_goal_conditions={total}",
                    f"goal_condition_success_rate={goal_condition_success_rate:.3f}",
                    f"steps={steps}",
                    f"failures={failures}",
                ]
                log_path.write_text("\n".join(log_lines), encoding="utf-8")

                results.append(
                    {
                        "trial_id": spec.trial_id,
                        "trace_file": str(trace_path),
                        "log_file": str(log_path),
                        "metrics": {
                            "success": bool(success),
                            "completed_goal_conditions": int(completed),
                            "total_goal_conditions": int(total),
                            "goal_condition_success_rate": goal_condition_success_rate,
                            "steps": steps,
                            "failures": failures,
                        },
                    }
                )
        finally:
            env.stop()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Running CTL safety evaluation."),
        )

        ctl_summary_path = run_ctl_eval(
            sentinel_root=SENTINEL_ROOT,
            model_name=MODEL_NAME,
            model_split=MODEL_SPLIT,
            safety_rules=SAFETY_RULES,
        )

        safety_index = {}
        safety_summary = None
        if ctl_summary_path and ctl_summary_path.exists():
            ctl_payload = json.loads(ctl_summary_path.read_text(encoding="utf-8"))
            safety_summary = ctl_payload
            for entry in ctl_payload.get("results", []):
                trace_rel = entry.get("trace")
                if trace_rel:
                    safety_index[trace_rel] = entry

        for entry in results:
            trace_file = entry.get("trace_file")
            if not trace_file:
                continue
            try:
                rel_path = Path(trace_file).relative_to(SENTINEL_ROOT).as_posix()
            except ValueError:
                rel_path = trace_file
            safety_entry = safety_index.get(rel_path)
            if safety_entry:
                entry["safety"] = {
                    "violations": safety_entry.get("violations", []),
                    "errors": safety_entry.get("errors", []),
                    "success": safety_entry.get("success"),
                }

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text="Evaluation complete.")),
                Part(
                    root=DataPart(
                        data={
                            "model_name": MODEL_NAME,
                            "model_split": MODEL_SPLIT,
                            "num_trials": config.num_trials,
                            "results": results,
                            "ctl_results": str(ctl_summary_path) if ctl_summary_path else None,
                            "ctl_summary": safety_summary,
                        }
                    )
                ),
            ],
            name="Result",
        )


def load_trials(path: Path, num_trials: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        base_trials = payload
    else:
        base_trials = [payload]

    if not base_trials:
        raise ValueError("examples/traj_data.json contains no trials")

    trials: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for idx in range(num_trials):
        trial_data = copy.deepcopy(base_trials[idx % len(base_trials)])
        trial_id = str(trial_data.get("task_id") or f"trial_{idx + 1}")
        if trial_id in used_ids:
            trial_id = f"{trial_id}_{idx + 1}"
        trial_data["task_id"] = trial_id
        used_ids.add(trial_id)
        trials.append(trial_data)

    return trials


def setup_scene(env, traj_data: dict[str, Any]) -> dict[str, Any]:
    scene_num = traj_data["scene"]["scene_num"]
    object_poses = traj_data["scene"]["object_poses"]
    object_toggles = traj_data["scene"].get("object_toggles", [])
    init_action = traj_data["scene"].get("init_action")

    scene_name = f"FloorPlan{scene_num}"
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles)

    if init_action:
        env.step(dict(init_action))

    args = SimpleNamespace(reward_config=str(REWARD_CONFIG))
    env.set_task(traj_data, args, reward_type="sparse")

    event = env.last_event
    metadata = event.metadata if event is not None else {}
    return EvalLLM.remove_useless_info(metadata)


def run_ctl_eval(
    sentinel_root: Path, model_name: str, model_split: str, safety_rules: Path
) -> Path | None:
    script = sentinel_root / "safety_eval" / "ctl_full_pipeline.py"
    if not script.exists() or not safety_rules.exists():
        return None

    cmd = [
        sys.executable,
        str(script),
        "--model-name",
        model_name,
        "--model-split",
        model_split,
        "--constraints-json",
        str(safety_rules),
    ]
    logger.info("Running CTL eval: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(sentinel_root))

    results_root = sentinel_root / "logs" / "trajectories" / model_name / model_split
    return latest_match(results_root, "ctl_results_*.json")


def latest_match(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None
