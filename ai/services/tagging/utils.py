from ai.schemas.tagging.tagging import TagRequest

# 태깅용 입력 문장 병합
def merge_content(req: TagRequest) -> str:
    return f"{req.situation} {req.choice} {req.outcome} {req.reflection}"