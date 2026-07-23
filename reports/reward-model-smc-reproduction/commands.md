# Reproduction command ledger

This ledger records the commands that changed project state or produced formal
evidence. Read-only `sed`, `find`, `rg`, and image-preview inspections are
omitted; their outputs did not alter evidence.

## Startup and source audit

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx skill orx-lit
orx skill orx-reports
orx skill report
orx projects --json
orx project view 2f60657b-6357-4bc7-9067-a2fd8bf63479
orx runs 2f60657b-6357-4bc7-9067-a2fd8bf63479
git branch -a
git rev-parse HEAD
git status --short
df -h .
orx paper 2602.01381
```

The paper HTML was retrieved from
`https://ar5iv.labs.arxiv.org/html/2602.01381` with an explicit browser
User-Agent, then hashed with `shasum -a 256`. The verdict dataset was fetched
from `ICML-2026-agent-repro/verdicts` and filtered strictly on
`space_id == "DineshAI/MrIDZjIsNF"`. The judged Space snapshot was downloaded
at exact revision `b675cbafc35867fc9212939818e54ff9225ac567` with:

```bash
hf download DineshAI/MrIDZjIsNF --repo-type space \
  --revision b675cbafc35867fc9212939818e54ff9225ac567 \
  --local-dir <temporary-read-only-snapshot>
```

No credential values or generated wrappers were printed.

## Frozen environment and baseline

```bash
uv lock
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 \
  --title "Frozen judged baseline" \
  --run-command "uv sync --frozen && .venv/bin/python repro/src/verify_smc.py"
git checkout orx/frozen-judged-baseline
git add .python-version pyproject.toml uv.lock
git commit -m "Pin reproducible uv environment"
git push origin HEAD
orx exp run dcabd6d8-215f-4ee4-8f03-2dec9ed1d88f --backend local
orx exp wait dcabd6d8-215f-4ee4-8f03-2dec9ed1d88f --timeout 480
orx logs b673402d-f18a-4708-aca2-bd06dceb4b3a
```

## Round 1: exact and statistical routes

```bash
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 \
  --title "Exact finite-state theorem harness" \
  --parent dcabd6d8-215f-4ee4-8f03-2dec9ed1d88f
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 \
  --title "Statistical scaling stress test" \
  --parent dcabd6d8-215f-4ee4-8f03-2dec9ed1d88f
git checkout orx/exact-finite-state-theorem-harness
uv run --frozen python -m py_compile repro/src/paper_models.py repro/src/verify_smc.py
git commit -m "Add exact finite-state theorem verification"
git push origin HEAD
orx exp run 877876b6-49f1-4f37-828d-1b7279dab0c8 --backend local
orx exp wait 877876b6-49f1-4f37-828d-1b7279dab0c8 --timeout 480
orx logs 907b6393-3ca9-43e1-88ac-06adf8b6304f
git checkout orx/statistical-scaling-stress-test
uv run --frozen python -m py_compile repro/src/statistical_models.py repro/src/verify_smc.py
git commit -m "Add independent statistical claim checks"
git push origin HEAD
orx exp run edb12195-8fb9-463e-9ce0-65e5170b4a58 --backend local
orx exp wait edb12195-8fb9-463e-9ce0-65e5170b4a58 --timeout 480
orx logs 3d60b792-f986-4474-9043-45b714887232
```

## Promoted MH and release-candidate runs

```bash
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 \
  --title "Cumulative evidence and resampling-pool MH" \
  --parent 877876b6-49f1-4f37-828d-1b7279dab0c8
git checkout orx/cumulative-evidence-and-resampling-pool-mh
uv run --frozen python -m py_compile repro/src/paper_models.py repro/src/verify_smc.py
git commit -m "Implement resampling-pool MH claim verifier"
git push origin HEAD
orx exp run 89e154c0-5cd8-4b1d-8465-a3d881914fbb --backend local
orx exp wait 89e154c0-5cd8-4b1d-8465-a3d881914fbb --timeout 480
orx logs 832b551d-d850-4e86-89f5-13ed1f3c3550
orx create-experiment 2f60657b-6357-4bc7-9067-a2fd8bf63479 \
  --title "Release-candidate cumulative evidence" \
  --parent 89e154c0-5cd8-4b1d-8465-a3d881914fbb
git checkout orx/release-candidate-cumulative-evidence
uv run --frozen python -m py_compile repro/src/paper_models.py repro/src/verify_smc.py
git commit -m "Generate release-candidate evidence package"
git push origin HEAD
orx exp run 0504c315-04fc-4f17-ba24-0762e830630e --backend local
orx exp wait 0504c315-04fc-4f17-ba24-0762e830630e --timeout 480
orx logs df1b7498-7f51-4f87-a660-ab6ebb1a1805 --range 0:30000
orx logs df1b7498-7f51-4f87-a660-ab6ebb1a1805 --range 30000:40000
```

Every formal node inherited this exact command:

```bash
uv sync --frozen && .venv/bin/python repro/src/verify_smc.py
```

## Release validation

The release-candidate command itself ran the SVG parser, `marimo check`, old/new
Space subset comparison, candidate-logbook JSON validation, text-only upload
allowlist generation, and generated-text secret scan. The publication copy was
then checked again with:

```bash
uv run --frozen marimo check --strict notebooks/reward_model_smc.py
git diff --check
git diff --no-index \
  <formal-run>/.openresearch/artifacts .openresearch/artifacts
git diff --no-index \
  <formal-run>/.openresearch/hf_upload .openresearch/hf_upload
git ls-remote origin
```

Hugging Face publication is intentionally absent: it requires explicit user
approval after the evidence forecast.
