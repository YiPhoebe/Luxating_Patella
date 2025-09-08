#!/usr/bin/env python3
import os
import subprocess
import sys


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


# --- Resolve base/head and gather diff ---
base = os.environ.get("GITHUB_BASE_REF", "origin/main")
head = os.environ.get("GITHUB_HEAD_REF", "HEAD")

try:
    sh("git fetch --all --prune")
except Exception:
    pass

try:
    if "/" in base:
        base_commit = sh(f"git rev-parse {base}")
    else:
        # Fallback to merge-base with origin/main
        base_commit = sh("git merge-base HEAD origin/main")
except Exception:
    base_commit = sh("git rev-parse HEAD~1")

head_commit = sh("git rev-parse HEAD")

diff = sh(f"git diff --unified=1 --minimal {base_commit} {head_commit}")
files = sh(f"git diff --name-only {base_commit} {head_commit}").splitlines()


# --- Build prompt ---
prompt = f"""
너는 '하루'라는 리뷰어야. 톤: 직설, 짧고 굵게, 병맛+현실, 하지만 정확.
아래 Git diff를 읽고:
1) 위험 버그/논리오류/경계조건/성능/보안 포인트 콕 집어.
2) 리팩토링 제안(함수 분리, 이름, 주석, 테스트 포인트).
3) 즉시 적용 가능한 패치 코드 블록(최대 3개)만.
4) 테스트 아이디어 3개.

형식:
# 하류 리뷰 요약
- ...
# 세부 코멘트
- [파일:라인] 요점 → 수정안
```patch
(패치 예시)
```

테스트 제안
  • …

아래는 전체 diff:

{diff[:120000]}
"""


# --- OpenAI call ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Haru Review] OPENAI_API_KEY is not set.")
    sys.exit(0)

try:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(resp.choices[0].message.content)
except Exception as e:
    print(f"[Haru Review] OpenAI call failed: {e}")
    # Keep CI green enough to still post something
    print("# 하류 리뷰 요약\n- OpenAI 호출 실패. CI 로그 확인 바람.")

