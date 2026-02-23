# Documentation Map

Use this page as the entrypoint to the project docs.

## Visual Map

```text
                           docs/index.md (you are here)
                                      |
             ---------------------------------------------------
             |                        |                        |
        Beginner Path           Builder Path             Deep Technical Path
             |                        |                        |
   docs/beginner_guide.md     docs/getting_started.md      docs/overview.md
             |                        |                        |
      docs/keywords.md        docs/architecture.md         docs/validation.md
             |                        |                        |
          docs/faq.md         docs/prerequisites.md        docs/original_proposal
```

## Reading Paths by Skill Level

### 1. New to Physics / Math / Data Science
1. `docs/beginner_guide.md`
2. `docs/keywords.md`
3. `docs/faq.md`
4. `docs/getting_started.md`

### 2. Comfortable with Python, New to This Project
1. `docs/getting_started.md`
2. `docs/architecture.md`
3. `docs/validation.md`
4. `docs/overview.md`

### 3. Technical Deep Dive (Research/Contributor)
1. `docs/overview.md`
2. `docs/architecture.md`
3. `docs/validation.md`
4. `docs/original_proposal`

## Doc Purpose Quick Reference

- `docs/beginner_guide.md`: Plain-language big picture.
- `docs/keywords.md`: Glossary for core terms used in this repository.
- `docs/faq.md`: Fast answers for common newcomer questions.
- `docs/getting_started.md`: Hands-on execution steps.
- `docs/architecture.md`: Module boundaries and code organization.
- `docs/overview.md`: Mathematical and algorithmic foundations.
- `docs/validation.md`: Reconstruction quality and testing metrics.
- `docs/prerequisites.md`: Background knowledge and learning resources.

## Suggested First Command After Reading

```bash
uv run --active python scripts/run_toy_2d.py --profile default
```

After running, compare your output with the example figure in `docs/assets/toy2d_summary_example.png`.
