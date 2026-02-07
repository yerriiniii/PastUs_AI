from ai.schemas.tagging import TagRequest

def merge_content(req: TagRequest) -> str:
    return f"{req.situation} {req.choice} {req.outcome} {req.reflection}"