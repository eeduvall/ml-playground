Review what happened in this session. Update .claude/skills/lstm-lessons/SKILL.md by:
- Appending any new architecture or hyperparameter findings
- Documenting failures with specific error conditions and values
- Updating "What To Try Next" based on what we learned
- Never deleting existing entries, only add or annotate them

Session notes: $ARGUMENTS
```

Then use it like:
```
/retrospective hidden_dim 512 caused NaN loss on MPS, switched to 256 and fixed it
