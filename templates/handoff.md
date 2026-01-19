# Agent Handoff

**From**: [Source Agent]
**To**: [Target Agent]
**Task Reference**: [task-ID]
**Timestamp**: YYYY-MM-DD HH:MM

## Context

[Brief context about the task and current state]

## Completed Work

- [x] [Completed item 1]
- [x] [Completed item 2]
- [x] [Completed item 3]

## Generated Files

| File | Description |
|------|-------------|
| `path/to/file1.py` | [Description] |
| `path/to/file2.py` | [Description] |

## Next Action

**Action Required**: [Clear description of what the next agent should do]

## Expected Output

- **Format**: [Code diff / New file / Updated file]
- **Location**: [Output path]
- **Success Criteria**: [How to know it's done correctly]

## Validation Method

```bash
# Command to verify the work
pytest tests/test_feature.py
```

## Known Issues / Warnings

- [Issue or caveat 1]
- [Issue or caveat 2]

## Context References

| Reference | Location | Purpose |
|-----------|----------|---------|
| Spec | `docs/spec.md#section-name` | Requirements |
| Plan | `.multi-llm/plans/plan.md` | Implementation plan |
| Previous code | `src/module/file.py:10-50` | Related implementation |

---

**Note**: Do not re-read entire files. Use the context references above.
