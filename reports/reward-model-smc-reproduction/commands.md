# Commands and provenance

## Fixed experiment command

This command was inherited unchanged by every node:

```bash
uv sync --frozen && .venv/bin/python repro/src/verify_smc.py
```

## Paper and judged-Space acquisition

```bash
orx paper 2602.01381 --full
hf download DineshAI/MrIDZjIsNF --repo-type space --revision 16f282752393f0d0b9a05950ff2a4ce57d7bbf8f --local-dir <temporary-directory>
```

The temporary download was read-only. Its 82 non-cache files were hashed into
the protected manifest committed on the winning branch.

## Experiment-tree commands

```bash
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 --title "Independent complexity and judge-visible v2" --parent 0504c315-04fc-4f17-ba24-0762e830630e
orx exp run a4cf5733-4c87-4129-81f3-e2d773fe3efe --backend local
orx exp wait a4cf5733-4c87-4129-81f3-e2d773fe3efe --timeout 480
orx logs 2c0ce216-932f-4b63-a784-40cae05497d2 --bytes 200000

orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 --title "Evaluator blind criticism audit" --parent a4cf5733-4c87-4129-81f3-e2d773fe3efe
orx exp run e422ecc6-a033-488c-a2a7-48c0e6040c5f --backend local
orx exp wait e422ecc6-a033-488c-a2a7-48c0e6040c5f --timeout 480
orx logs 5882470e-ab8c-4432-a3b1-67fe4df44f4a --bytes 100000
```

## Static and release-gate checks

```bash
.venv/bin/python -m py_compile repro/src/paper_models.py repro/src/judge_visible_v2.py repro/src/verify_smc.py
git diff --check
git ls-remote origin refs/heads/orx/evaluator-blind-criticism-audit refs/heads/master
orx runs 2f60657b-6357-4bc7-9067-a2fd8bf63479
```

The fixed verifier itself performed the SVG parse checks, `marimo check`,
claim verifiers, independent checkers, negative controls, JSON generation,
secret scan, immutable-Space subset audit, and evaluator-visibility audit.

## Publication and remote verification

```bash
hf auth whoami
hf spaces info DineshAI/MrIDZjIsNF --revision main --format json
hf download DineshAI/MrIDZjIsNF --repo-type space --revision 16f282752393f0d0b9a05950ff2a4ce57d7bbf8f --local-dir <preflight-directory>
/opt/homebrew/Cellar/hf/1.24.0/libexec/bin/python <text-only-create-commit-script>
hf spaces info DineshAI/MrIDZjIsNF --revision main --format json
hf download DineshAI/MrIDZjIsNF --repo-type space --revision e646b236a4ba1e68b5bc246fb48a2d9f6113e4dd --local-dir <verification-directory>
git fetch origin master
git push origin HEAD:master
git ls-remote origin refs/heads/master
hf download ICML-2026-agent-repro/verdicts --repo-type dataset --include verdicts.json --local-dir <verdict-directory>
```

The publication script decoded every payload as UTF-8, rechecked every
allowlist hash, created 91 `CommitOperationAdd` operations and zero delete
operations, and pinned `parent_commit` to
`16f282752393f0d0b9a05950ff2a4ce57d7bbf8f`. It did not read or print a token.
The verdict dataset was filtered only by
`space_id == "DineshAI/MrIDZjIsNF"`.
