# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

이 파일은 Claude Code가 본 저장소에서 작업할 때 참고하는 가이드입니다.

=======================================================================================

> **팀 프로젝트 주의사항** — 이 파일은 팀과 공유됩니다.
> 작업이 끝난 뒤 **`git push` 하기 전에 항상 이 CLAUDE.md를 최신 상태로 업데이트**해 주세요.
> (새 모듈 추가, CLI 인자 변경, 학습 파이프라인 수정 등 반영)

## 1. Git 커밋/푸쉬 규칙 (팀 합의)

> **❗ 매우 중요 — Claude Code 사용 시**
>
> - 커밋 메시지에 **`Co-Authored-By: Claude ...` 줄(co-author 트레일러)을 절대 포함하지 말 것.**
>   기본 동작에서 추가되는 `Co-Authored-By: Claude Opus ...` 같은 라인은 모두 **제거하고** 커밋한다.
> - 마찬가지로 `🤖 Generated with [Claude Code]` 같은 자동 서명 푸터도 넣지 않는다.
> - 즉, 커밋 메시지는 **사람이 직접 쓴 것처럼 본문(제목 + 설명)만** 남긴다.

### 커밋 예시 (OK)
```
feat(losses): CB-Focal 가중치 정규화 방식 변경

weight.sum() == num_classes 가 되도록 정규화하여
official impl 과 동치가 되게 수정.
```

### 커밋 예시 (NG — 이렇게 쓰지 말 것)
```
feat(losses): CB-Focal 가중치 정규화 방식 변경

...

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

### 푸쉬 전 체크리스트
1. 변경된 코드/명령/파이프라인을 **CLAUDE.md (이 파일) 와 README.md** 에 반영했는가?
2. 커밋 메시지에 `Co-Authored-By:` 또는 Claude 자동 서명이 들어가 있지 않은가?
3. `outputs/`, `data/`, `*.pth`, 큰 캐시 파일 등을 실수로 커밋하지 않았는가? (필요하면 `.gitignore` 확인)
4. (가능하면) 학습 한 epoch 라도 돌려서 import / dataloader / loss 가 깨지지 않는지 확인.
