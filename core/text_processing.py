"""
text_processing.py — 텍스트 정제 유틸

크롤링 결과에서 AI 분석에 적합한 핵심 텍스트 추출.
"""

import re


def extract_business_text(raw_text: str, max_len: int = 3000) -> str:
    """
    크롤링 원본 텍스트에서 비즈니스 분석용 핵심 텍스트 추출.
    - 불필요한 공백/특수문자 제거
    - 너무 짧은 라인 제거
    - max_len 글자로 제한
    """
    if not raw_text:
        return ""

    # 연속 공백/개행 정리
    text = re.sub(r'\s{3,}', ' ', raw_text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 의미없는 짧은 토큰 제거 (1~2글자만 있는 라인)
    lines = []
    for ln in text.split('\n'):
        ln = ln.strip()
        if len(ln) >= 4:
            lines.append(ln)

    clean = ' '.join(lines)

    # HTML 엔티티 제거
    clean = re.sub(r'&[a-zA-Z]+;', ' ', clean)
    clean = re.sub(r'&#\d+;', ' ', clean)

    # 연속 공백 정리
    clean = re.sub(r'\s{2,}', ' ', clean).strip()

    return clean[:max_len]
