Evaluate the model checkpoint at $ARGUMENTS.
Run on the validation set, print top-5 suggestions for 10 sample queries,
and report BLEU and perplexity scores.
```

---

### Step 6: Session Workflow Tips

Use `/model` to select model and reasoning, `/context` to see context usage, and `/usage` to check plan limits. Always use thinking mode and explanatory output style for better understanding of Claude's decisions. Use the `ultrathink` keyword in prompts for high-effort reasoning. 

Key habits to adopt:
- Use `/clear` between unrelated tasks to avoid polluting context with irrelevant information. 
- Manually run `/compact` at around 50% context usage to avoid the "agent dumb zone." 
- Add to your `CLAUDE.md`: `When compacting, always preserve: current file being edited, training run config, architecture decisions made this session.`

---

### Starting Prompt for Your LSTM Project

Once configured, kick things off with a scoped, well-defined prompt like:
```
ultrathink — scaffold a PyTorch LSTM search suggester. 
Architecture: encoder LSTM over tokenized query prefixes → decoder that 
generates top-k completions. Include:
- Model class in src/models/lstm_suggester.py
- Dataset class in src/data/query_dataset.py
- Training loop in src/train/train.py with config YAML
- Inference function returning ranked suggestions
Use type hints, docstrings, and no hardcoded hyperparameters.