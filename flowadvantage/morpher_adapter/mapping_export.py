"""Export versioned Morpher bridge documents, without fabricating routes."""
import json

CANONICAL_SCHEMA = "flowadvantage_morpher_mapping_v1"


def export_flowadvantage_mapping(mapping, path=None):
    """Emit the native bridge schema from an already concrete mapping.

    A FlowAdvantage search result must provide native resource IDs and ordered
    route resources.  This intentionally rejects the old ``assignments``-only
    form because Morpher cannot safely replay it.
    """
    if not isinstance(mapping, dict):
        raise TypeError("mapping must be a dictionary")
    required = ("architecture_hash", "dfg_hash", "ii", "operations", "routes")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError("canonical Morpher mapping is missing " + ", ".join(missing))
    out = {
        "schema": CANONICAL_SCHEMA,
        "toolchain_commit": mapping.get("toolchain_commit", "unknown"),
        "architecture_hash": mapping["architecture_hash"],
        "dfg_hash": mapping["dfg_hash"],
        "ii": mapping["ii"],
        "operations": mapping["operations"],
        "dependencies": mapping.get("dependencies", []),
        "resources": mapping.get("resources", []),
        "routes": mapping["routes"],
        "memory_bindings": mapping.get("memory_bindings", []),
        "metadata": mapping.get("metadata", {}),
    }
    if path:
        with open(path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    return out


def export_mapping(mapping, path=None):
    """Backward-compatible fixture exporter used by historic unit tests."""
    if isinstance(mapping, dict) and mapping.get("schema") == CANONICAL_SCHEMA:
        return export_flowadvantage_mapping(mapping, path)
    out = {"format": "morpher_mapping_v1", "assignments": mapping.get("assignments", mapping) if isinstance(mapping, dict) else mapping}
    if path:
        with open(path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    return out
