# scripts/haru_review.py
# 출력: review.md (stdout) — GitHub Actions에서 이 파일을 코멘트로 달아줍니다.
import subprocess
import sys
from textwrap import dedent


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def try_get_base() -> str:
    # PR 컨텍스트면 main과의 공통 조상, 아니면 직전 커밋
    try:
        return sh("git merge-base HEAD origin/main")
    except subprocess.CalledProcessError:
        return sh("git rev-parse HEAD~1")


def changed_files(base: str, head: str) -> list[tuple[str, str]]:
    """
    반환: [(status, path), ...]  예) 'M', 'A', 'D'
    """
    out = sh(f"git diff --name-status {base}..{head}")
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def render_review(changes: list[tuple[str, str]]) -> str:
    added = [p for s, p in changes if s.upper().startswith("A")]
    modified = [p for s, p in changes if s.upper().startswith("M")]
    deleted = [p for s, p in changes if s.upper().startswith("D")]
    renamed = [p for s, p in changes if s.upper().startswith("R")]

    files_md: list[str] = []
    if added:
        files_md.append("**추가됨**\n" + "\n".join(f"- {p}" for p in added))
    if modified:
        files_md.append("**수정됨**\n" + "\n".join(f"- {p}" for p in modified))
    if deleted:
        files_md.append("**삭제됨**\n" + "\n".join(f"- {p}" for p in deleted))
    if renamed:
        files_md.append("**이름 변경**\n" + "\n".join(f"- {p}" for p in renamed))
    if not files_md:
        files_md = ["(변경 파일 없음)"]

    rename_line = f", 이름 변경: **{len(renamed)}**" if renamed else ""
    summary_line = f"- 추가: **{len(added)}**, 수정: **{len(modified)}**, 삭제: **{len(deleted)}**{rename_line}"

    # ⚠️ f-string 안에서 백슬래시 사용 금지 → 미리 조립
    nl = "\n"
    files_block = nl.join(files_md)

    body = dedent(
        f"""
        ## 🤖 하루 자동 리뷰

        ### 요약
        - 이 PR에는 총 **{len(changes)}개 파일**의 변경이 있습니다.
        {summary_line}

        ### 변경된 파일 목록
        {files_block}

        ### 체크리스트 (작성자 확인용)
        - [ ] 테스트 코드가 있나요?
        - [ ] 주요 함수/클래스에 주석(docstring)을 추가했나요?
        - [ ] 린트/타입 오류를 해결했나요?

        ---
        _이 코멘트는 `scripts/haru_review.py`에서 자동으로 생성되었습니다._
        """
    ).strip()

    return body


def main():
    base = try_get_base()
    head = sh("git rev-parse HEAD")
    changes = changed_files(base, head)
    print(render_review(changes))


if __name__ == "__main__":
    sys.exit(main())
