#!/usr/bin/env python3
import json, sys

path = sys.argv[1]
with open(path) as f:
    state = json.load(f)

print(f"total_steps: {state.get('global_step')}")
print(f"best_metric: {state.get('best_metric')}")
print(f"best_step: {state.get('best_global_step')}")
print(f"epoch: {state.get('epoch')}")
print()
for entry in state['log_history'][-10:]:
    print(entry)
