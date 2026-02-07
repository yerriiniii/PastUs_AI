import os
import json
import random
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

client = OpenAI(api_key=api_key)


# 전체 태그 목록
ALL_TAGS = [
    "군대", "휴학/복학", "성적", "전과", "연애",
    "장학금", "과제/팀플", "대학원", "졸업",
    "새내기", "자취", "편입", "N수/반수", "취업"
]

# 타겟 태그 목록
TARGET_TAGS = [
    "군대", "휴학/복학", "장학금",
    "대학원", "편입", "N수/반수", "취업"
]

# 태그별 상황 정의 
TAG_DEFINITIONS = {
    "군대": "입대, 군 복무, 전역 등 군 생활과 관련된 결정이나 고민",
    "휴학/복학": "학업을 중단하거나 다시 시작하는 것과 관련된 결정이나 고민",
    "성적": "시험, 학점, 평가 결과로 인한 고민이나 결과",
    "전과": "기존 전공을 변경하는 결정이나 그에 따른 고민이나 결과",
    "연애": "연인, 썸, 이별 등 연애 관계에서의 감정에 따른 고민이나 선택",
    "장학금": "성적 또는 조건에 따라 장학금을 받기 위한 노력이나 결과",
    "과제/팀플": "과제, 팀 프로젝트, 조별 활동에서 발생한 경험이나 갈등",
    "대학원": "석사 또는 박사 과정 진학, 연구실, 연구 주제에 대한 고민이나 선택",
    "졸업": "졸업 요건, 졸업 시점, 졸업 이후 진로에 대한 고민이나 결과",
    "새내기": "대학 입학 초기 적응 과정에서의 경험이나 고민",
    "자취": "자취, 독립 생활, 주거 환경 변화로 인한 고민이나 경험",
    "편입": "다른 대학으로 편입을 준비하거나 결정하기 위한 고민이나 결정",
    "N수/반수": "수능을 다시 준비하거나 재도전하고싶은 고민이나 경험",
    "취업": "회사 지원, 면접, 합격 또는 불합격과 관련된 진로 고민이나 결과"
}


# GPT 샘플 생성 프롬프트
def build_prompt(datas_num, forced_target):
    return f"""
너는 ‘정답이나 충고보다는, 
비슷한 상황에 있던 사람들이 과거에 실제로 어떤 선택을 했고
시간이 지난 뒤 그 선택을 어떻게 다시 바라보게 되었는지’를
사례로 공유하는 서비스 'PastUs'에 올라갈
과거 경험 공유 사례를 생성하는 역할이야.

이 데이터는
사용자 게시글의 ‘고민 상황 태그 분류 AI 모델 학습용’으로 사용된다.

아래 포맷을 반드시 모두 사용해
하나의 과거 경험 사례를 작성하라.

---

[상황 설명]
- 언제, 어떤 고민 상황이었는지 서술
- 당시의 맥락과 배경이 드러나야 함
- 분량: 50~300자

[당시 선택]
- 실제로 어떤 선택을 했는지
- 왜 그 선택을 하게 되었는지
- 분량: 50~300자

[결과]
- 그 선택 이후 실제로 어떤 일이 있었는지
- 긍정/부정 결과가 구체적인 사건이나 변화로 드러나야 함
- 분량: 50~300자

[지금의 생각]
- 시간이 지난 지금, 그 선택을 어떻게 평가하는지
- 다시 같은 상황이라면 같은 선택을 할지, 아닐지가 드러나야 함
- 판단 기준이나 관점의 변화 포함
- 분량: 50~300자

---

공통 조건:
- 반드시 이미 겪은 과거 경험이어야 함 (회고 시점)
- 단순한 감정 기록이나 일기처럼 작성하지 말 것
- 이 글은 ‘지금 비슷한 고민을 하는 사람에게 참고가 되는 사례’ 전제로 작성
- 충고, 훈수, 해결책 제시, 질문 형태의 문장 금지
- 문어체 사용, 줄임말·가벼운 오타 허용
- 태그는 반드시 게시글의 고민 내용에서 직접적으로 드러나는 상황만 선택
- 추측이나 연상으로 태그를 붙이지 말 것
- 필요하면 1개, 2개, 3개 이상 등 개수 제한 없이 맥락과 관련있는 태그를 정확하게 선택
- 사례 간 내용, 표현, 상황이 최대한 겹치지 않게 작성
- 태그가 한쪽으로 치우치지 않도록 사례를 다양하게 생성

---

태그 목록:

주 타겟 태그 (이번 생성에서 반드시 중심 주제로 사용): 
{forced_target}

보조 태그 (상황과 직접적으로 관련되어 있을 시 선택 가능): 
{ALL_TAGS}

---

총 {datas_num}개의 사례를 생성하라.
출력은 반드시 JSON 배열만 사용하고, 
설명, 주석, 코드블록은 절대 사용하지 말 것.
각 사례는 아래 형식을 그대로 따를 것:

[
  {{
    "text": {{
      "situation": "...",
      "choice": "...",
      "outcome": "...",
      "reflection": "..."
    }},
    "tag": ["태그1", "태그2", "..."]
  }}
]
"""

# GPT 태그 의미 검증 프롬프트
def build_validation_prompt(text_block, tag):
    text = " ".join(text_block.values())
    tag_def = TAG_DEFINITIONS[tag]

    return f"""
다음 글이 아래 태그 상황을
**실제로 핵심적으로 다루고 있는지** 판단하라.

[태그]
{tag}: {tag_def}

[글]
{text}

답변은 반드시 다음 중 하나만 출력:
YES 또는 NO
"""


# GPT 호출 함수 -> 샘플 생성
def generate_samples(datas_num, forced_target):
    prompt = build_prompt(datas_num, forced_target)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


# GPT 호출 함수 -> 태그 의미 검증
def is_tag_semantically_valid(tag, text_block):
    prompt = build_validation_prompt(text_block, tag)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return answer == "YES"


# 메인 루프
def main():
    TARGET_TOTAL = 400
    SAMPLES_PER_BATCH = 10

    all_datas = []
    pbar = tqdm(total=TARGET_TOTAL, desc="유효 샘플 수집")

    while len(all_datas) < TARGET_TOTAL:
        forced_target = random.choice(TARGET_TAGS)
        batch = generate_samples(SAMPLES_PER_BATCH, forced_target)

        for data in batch:
            if "text" not in data or "tag" not in data:
                continue

            raw_tags = [t for t in data["tag"] if t in ALL_TAGS]

            # 타겟 태그 포함 여부
            if forced_target not in raw_tags:
                continue

            validated_tags = []
            for tag in raw_tags:
                if is_tag_semantically_valid(tag, data["text"]):
                    validated_tags.append(tag)

            # 타겟 태그가 의미 검증에서 탈락하면 폐기
            if forced_target not in validated_tags:
                continue

            data["tag"] = validated_tags
            all_datas.append(data)
            pbar.update(1)

            if len(all_datas) >= TARGET_TOTAL:
                break

    pbar.close()

    # 태그 분포 출력
    tag_counts = defaultdict(int)
    for d in all_datas:
        for t in d["tag"]:
            tag_counts[t] += 1

    print("\n태그 분포:")
    for k, v in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")

    # 저장
    os.makedirs("data", exist_ok=True)
    path = "data/tag_focus_dataset_5.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_datas, f, ensure_ascii=False, indent=2)

    print("\n생성 완료")
    print(f"총 사례 수: {len(all_datas)}")
    print(f"저장 경로: {path}")


if __name__ == "__main__":
    main()
