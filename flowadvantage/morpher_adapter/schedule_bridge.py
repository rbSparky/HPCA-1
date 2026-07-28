def schedule_window(node):
    return (int(node.get("asap",0)), int(node.get("alap",node.get("asap",0))))
