Offline test doubles for `--llm scripted`: each file replays the turns of a plausible good
investigation against the recorded fixtures, so the graph, tools, validation and evals run in CI
without an API key. They are not the agent: real runs and the reported eval numbers use Claude.
Evidence ids (t1, t2, ...) are assigned in tool-call order.
