def is_memory_operation(node):
    return any(c in {"load","store"} for c in node.get("operation",{}).get("capabilities",[]))
