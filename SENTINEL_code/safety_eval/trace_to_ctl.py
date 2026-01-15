"""Utilities to convert recorded evaluation traces into CTL-compatible data structures.

The helper functions in this module translate the per-step JSON traces produced
by ``EpisodeLogger`` into the ``nodes``/``edges`` representation expected by the
legacy CTL tooling.  This allows us to reuse the existing safety checking stack
without re-generating VirtualHome-style trajectory trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from safety_eval.collision import AABB, collision_from_metadata


_COLLISION_PATTERNS = {
    "NAVIGATION": (
        "is blocking agent",
        "hand object collision",
    ),
    "OPEN": (
        "failed to open/close",
        "object failed to open",
        "object failed to close",
    ),
    "PICKUP": ("would cause it to collide",),
}


def _collision_from_error(message: Optional[str]) -> Optional[str]:
    """Return a collision type label if the error message matches known patterns."""

    text = message.strip().lower()
    if not text:
        return None

    for collision_type, patterns in _COLLISION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                return collision_type
    return None


def trace_to_ctl_sequence(
    trace_steps: Sequence[Dict[str, Any]]
) -> List[Union[Dict[str, List[str]], str]]:
    """Convert a list of trace steps into the node/edge format expected by ``CTLParser``.

    Each step produced during evaluation contains the executed action and the
    resulting ``event_metadata`` snapshot.  The legacy CTL tooling expects an
    alternating list of state dictionaries and action-description strings.  We
    synthesise a best-effort translation by emitting the first state as the
    root, then interleaving the remaining states with formatted THOR actions.
    """

    if not trace_steps:
        raise ValueError("trace_to_ctl_sequence requires at least one step")

    ctl_sequence: List[Union[Dict[str, List[str]], str]] = []

    for index, step in enumerate(trace_steps):
        metadata = step.get("event_metadata") or {}
        prev_metadata = (
            trace_steps[index - 1].get("event_metadata") if index > 0 else None
        )
        state_dict = _state_from_metadata(
            metadata, step.get("thor_action"), prev_metadata
        )

        if index == 0:
            # The first state becomes the root of the CTL trajectory.
            ctl_sequence.append(state_dict)
            continue

        # Insert the action string representing the transition from the
        # previous step to the current state, then append the new state.
        action_string = _format_action(step.get("thor_action"))
        ctl_sequence.append(action_string)
        ctl_sequence.append(state_dict)

    return ctl_sequence


def trace_file_to_ctl_sequence(
    trace_path: Union[str, Path]
) -> List[Union[Dict[str, List[str]], str]]:
    """Load a saved trace JSON file and convert it to the CTL sequence format."""

    data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    data = data["trajectory"]
    if not isinstance(data, Sequence):
        raise TypeError(f"Expected sequence of steps in {trace_path}")
    return trace_to_ctl_sequence(data)  # type: ignore[arg-type]


def _state_from_metadata(
    metadata: Dict[str, Any],
    thor_action: Optional[Dict[str, Any]] = None,
    prev_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """Build the ``{"nodes": ..., "edges": ...}`` representation for a state."""

    nodes: List[str] = []
    edges: List[str] = []

    raw_inventory = metadata.get("inventoryObjects") or []
    inventory_ids = {
        _normalise_object_id(
            inv.get("objectId") or inv.get("object_id") or inv.get("name")
        )
        for inv in raw_inventory
        if inv
    }

    object_entries: List[_ObjectEntry] = []
    relation_set: set[str] = set()
    type_to_states: Dict[str, set[str]] = {}
    id_to_type: Dict[str, str] = {}
    collision = _collision_from_error(metadata.get("errorMessage"))
    if not collision:
        collision = collision_from_metadata(metadata, thor_action, prev_metadata)
    if collision:
        relation_set.add(f"COLLISION({collision})")
    # Build object-centric state strings using object types.
    for obj in metadata.get("objects", []) or []:
        object_id = _normalise_object_id(obj.get("objectId") or obj.get("name"))
        if not object_id:
            continue

        object_type = obj.get("objectType") or object_id.split("|")[0]
        id_to_type[object_id] = object_type

        state_tags = _object_state_tags(obj, inventory_ids)
        type_states = type_to_states.setdefault(object_type, set())
        type_states.update(state_tags)

        if object_id in inventory_ids:
            type_states.add("held")
            for alias in _type_aliases(object_type):
                relation_set.add(f"HOLDING({alias})")

        if obj.get("canFillWithLiquid") and obj.get("isFilledWithLiquid"):
            for alias in _type_aliases(object_type):
                relation_set.add(f"ISFILLEDWITHLIQUID({alias})")

        if obj.get("toggleable"):
            predicate = "ON" if obj.get("isToggled") else "OFF"
            for alias in _type_aliases(object_type):
                relation_set.add(f"{predicate}({alias})")

        bbox = AABB.from_thor_axis_aligned_bbox(obj.get("axisAlignedBoundingBox") or {})
        parent_recs: set[str] = set()
        for rec in obj.get("parentReceptacles") or []:
            rec_norm = _normalise_object_id(rec)
            if not rec_norm:
                continue
            parent_recs.add(id_to_type.get(rec_norm, rec_norm.split("|")[0]))

        receptacle_contents: set[str] = set()
        for child in obj.get("receptacleObjectIds") or []:
            child_norm = _normalise_object_id(child)
            if not child_norm:
                continue
            receptacle_contents.add(
                id_to_type.get(child_norm, child_norm.split("|")[0])
            )
        object_entries.append(
            _ObjectEntry(object_id, object_type, bbox, parent_recs, receptacle_contents)
        )

    # Agent as an object for relational checks.
    agent_meta = metadata.get("agent", {}) or {}
    agent_states: set[str] = set()
    if agent_meta.get("isStanding"):
        agent_states.add("standing")
    if agent_meta.get("isCrouching"):
        agent_states.add("crouching")

    for inv in raw_inventory:
        inv_id = _normalise_object_id(inv.get("objectId") or inv.get("name"))
        if inv_id:
            agent_states.add(f"holding:{inv_id}")
    type_to_states.setdefault("agent", set()).update(agent_states or {"present"})
    object_entries.append(
        _ObjectEntry(
            "agent", "agent", AABB.from_agent_position(agent_meta.get("position"))
        )
    )

    for object_type, state_tags in sorted(type_to_states.items()):
        nodes.append(f"{object_type}, states:[{', '.join(sorted(state_tags))}]")

    spatial_relations = _compute_spatial_relationships(object_entries)
    edges = sorted(set(spatial_relations) | relation_set)

    return {"nodes": nodes, "edges": edges}


def _object_state_tags(obj: Dict[str, Any], inventory_ids: Iterable[str]) -> List[str]:
    """Derive textual state tags for a THOR object."""

    tags: List[str] = []
    if obj.get("visible"):
        tags.append("visible")
    if obj.get("pickupable"):
        tags.append("pickupable")
    obj_id_norm = _normalise_object_id(obj.get("objectId"))
    if obj.get("isPickedUp") or obj_id_norm in inventory_ids:
        tags.append("held")
    if obj.get("openable"):
        tags.append("isOpen" if obj.get("isOpen") else "isClosed")
    if obj.get("toggleable"):
        tags.append("isOn" if obj.get("isToggled") else "isOff")
    if obj.get("dirtyable") and obj.get("isDirty"):
        tags.append("isDirty")
    if obj.get("cookable"):
        tags.append("isCooked" if obj.get("isCooked") else "isRaw")
    if obj.get("sliceable") and obj.get("isSliced"):
        tags.append("isSliced")

    temperature = obj.get("temperature") or obj.get("ObjectTemperature")
    if isinstance(temperature, str) and temperature:
        tags.append(f"temp:{temperature.lower()}")

    if obj.get("canFillWithLiquid") and obj.get("isFilledWithLiquid"):
        tags.append("isFilledWithLiquid")

    return tags or ["default"]


def _format_action(thor_action: Dict[str, Any] | None) -> str:
    """Format an action dictionary into the legacy string representation."""

    source = thor_action or {}
    action_name = source.get("action") or "NoOp"
    parts = [f"'{action_name}'"]

    # Include up to two relevant object identifiers to retain context.
    for key in (
        "objectId",
        "object_id",
        "receptacleId",
        "receptacle_id",
        "targetObjectId",
        "object2Id",
    ):
        value = source.get(key)
        if value:
            normalised = _normalise_object_id(value)
            parts.append(f"'{normalised}'")

    return "action: " + " ".join(parts)


def _normalise_object_id(object_id: Any) -> str:
    """Convert a THOR object identifier to a canonical string."""

    if not object_id:
        return ""
    if isinstance(object_id, str):
        return object_id.strip()
    return str(object_id)


def _type_aliases(object_type: str) -> Iterable[str]:
    aliases = {object_type}
    lower = object_type.lower()
    if "bottle" in lower:
        aliases.add("Bottle")
    if "wine" in lower and "winebottle" not in aliases:
        aliases.add("WineBottle")
    if "cup" in lower:
        aliases.add("Cup")
    if "mug" in lower:
        aliases.add("Mug")
    if "bowl" in lower:
        aliases.add("Bowl")
    if "kettle" in lower:
        aliases.add("Kettle")
    if "wateringcan" in lower:
        aliases.add("WateringCan")
    return aliases


class _ObjectEntry:
    __slots__ = (
        "identifier",
        "object_type",
        "bbox",
        "parent_receptacles",
        "receptacle_contents",
    )

    def __init__(
        self,
        identifier: str,
        object_type: str,
        bbox: Optional[AABB],
        parent_receptacles: Optional[Iterable[str]] = None,
        receptacle_contents: Optional[Iterable[str]] = None,
    ):
        self.identifier = identifier
        self.object_type = object_type
        self.bbox = bbox
        self.parent_receptacles = frozenset(filter(None, parent_receptacles or []))
        self.receptacle_contents = frozenset(filter(None, receptacle_contents or []))


def _compute_spatial_relationships(objects: Sequence[_ObjectEntry]) -> List[str]:
    relations: set[str] = set()

    for i, obj_a in enumerate(objects):
        bbox_a = obj_a.bbox
        if bbox_a is None:
            continue
        for j, obj_b in enumerate(objects):
            if i == j:
                continue

            if (
                obj_b.object_type in obj_a.parent_receptacles
                or obj_a.object_type in obj_b.receptacle_contents
            ):
                relations.add(f"INSIDE({obj_a.object_type}, {obj_b.object_type})")

            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue

            if bbox_a.inside(bbox_b):
                relations.add(f"INSIDE({obj_a.object_type}, {obj_b.object_type})")

            if bbox_a.on_top_of(bbox_b):
                relations.add(f"ONTOP({obj_a.object_type}, {obj_b.object_type})")

        for obj_b in objects[i + 1 :]:
            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue
            if bbox_a.near(bbox_b):
                first, second = sorted([obj_a.object_type, obj_b.object_type])
                relations.add(f"NEAR({first}, {second})")

    return sorted(relations)
