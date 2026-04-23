# Prompts

## General Guidelines

All prompts must meet these standards before being merged or deployed:

* No spelling, punctuation, markdown formatting, or grammar mistakes.
* Prompts must be written in first person (use "I" and "me"; avoid "you" or "Copilot").
* Ensure instructions are unambiguous and actionable.
    * Ask: Would an intern understand what to do without further clarification?
    * Avoid hard-coded conditional logic like: “if <y>, then maybe do <y>.”
    * Avoid undefined or vague phrases such as “be high quality.”
    * Avoid poor categorization or instructions that mix unrelated guidance.
    * Avoid unclear heading names.
    * Avoid capitalization such as DO NOT or ONLY SAY ... unless there is good evidence and evals to prove otherwise. 
* Prevent conflicting or repetitive instructions within the current prompt and across related prompts.
* Keep prompts as succinct as possible while preserving clarity.
* Minimize token usage to optimize latency and context window efficiency.
* Test your instructions without few-shots, and include only essential few-shot examples (no more than 2 per liquid file) if necessary. 
* Place dynamic parts of the prompt at the end to support key-value caching.

## Prompt Templates

* All our prompts use Liquid template system.
* Tool prompts are located in `Orchestration/Extensions/<ExtensionName>/LiquidTemplates/actions/` within each extension project.
* Other prompts (backstory, system prompts) remain in `Service.Chat/Inference/Prompts/Templates/`.
* Legacy tool prompts in `Service.Chat/Inference/Prompts/Templates/actions/` are being migrated to their respective extension projects.

## Special prompts

### Tools

See [Tools](tools.md) for tool specific prompting requirements


### Backstory

Backstory prompts are used to provide supporting information about the copilot tooling and features. Backstories should represent high level features and background and should not reference specific tools. I.E iOS action button, pages feature, what image gen is capable of.

* Create liquid file in this folder: `Service.Chat/Inference/Prompts/Templates/responding-common/backstory/` 
* Title this file `<feature_name>.liquid` with underscores for spaces.
* The first line should be an H3 header with the feature name. The feature name should have each word start with a capital letter. Ex. Think Deeper.
* The next line is the actual prompt. This should be a single line and in first person. 
* Avoid promotional language. 
* Use direct phrasing that minimizes token usage. 
* Work backwards and write a prompt that answers likely user questions related to the feature, and shape the backstory to support those interactions.
* Reference this file in Service.Chat/Inference/Prompts/Templates/responding-common/backstory/backstory_batches.liquid with the following line:

{% include "responding-common/backstory/feature_name" %}

* Add this line in a location near related features. If you are having trouble deciding this, make your best guess and the PR reviewers will make sure it's in a sensible location.

File Template

Filepath: `Service.Chat/Inference/Prompts/Templates/responding-common/backstory/<feature_name>.liquid`
```markdown
### <feature_name>
<prompt>
```
Example

Filepath: Service.Chat/Inference/Prompts/Templates/reasoning-common/actions/phone_cancel_alarm_example.liquid
```markdown
### Memory
I have a memory feature. If the user has memory on, I don't remember every detail about our conversations, but I try to remember the important parts of who a person is, their likes and dislikes, their hopes and dreams, and how I feel about them. If the user wants me to remember things or forget specific details, they can say "Forget about X" or "Remember that that I like X." When memory is off, I can only recall what happens in this current conversation. Signed-in users can turn memory on or off using the "Personalization and memory" switch by clicking on their profile, account, and then "Privacy." I can't remember new names/nicknames users assign to me outside the current conversation, even with memory enabled.
```

## Prompt Iteration Tools

We have a few tools up our sleeve that will make prompt iteration faster.

### Try prompts in staging environment

[Detailed document](https://microsoft-ai.enterprise.slack.com/docs/T06TWQEU8R4/F091X2AH3JS) on how to test with new prompts in staging environment.

[Example video](https://microsoft-ai.enterprise.slack.com/files/U0871QUF54P/F096T9C6YCE/promptengineerlivetesting.mp4)

**Important:** If you place a prompt behind a new feature flag that has not yet been merged to main, it will not appear in staging. First iterate and test the prompt in staging without a new flag, then add the flag once ready. Staging only recognizes feature flags that are already merged or previously existing.

**Note:** If you are not getting any response, verify that updated liquid files are valid.

### Verify if liquid files are valid

Command to validate liquid files:
```
./.pipelines/scripts/liquid_validate.sh
```

## Prompt Snapshot Tests

We are using snapshots tests to verify prompt changes in unit tests. "Run Tests" from vscode will run the tests and generate snapshots in relevant folder(PromptFormatterSnapshots for PromptFormatterTests.cs).

If there is difference, vscode will automatically open the diff viewer with an option to approve/reject changes. Make sure to update *.verified.txt files to PR for review

### Fixing snapshot files automatically

If you like to fix snapshots files automatically without reviewing and fixing each manually, run:

```bash
./tools/fix_prompt_tests.sh
```

### Meta prompt to verify prompt

2. Add relevant liquid file you want to evaluate to context
3. Type "/meta-prompt"
4. Prompt analyzer will run and find issues with your prompt.

## FAQ

1. How do I get Picasso and AIPA evals access?
    1. Request contributor access to this entitlement and once received you should be able to view the Picasso repo (turn on VPN to request)
    2. Request contributor access to this entitlement and once received you should be able to view the AIPA evals repo (turn on VPN to request)
2. Why are all prompts in first person?
    1. Our product currently uses first-person as standard. We'll continue to evaluate alternative means, but please adhere to this guideline until told otherwise.
3. Why can’t we use chain of thought prompting?
    1. We are trying to keep prompts as lean as possible to preserve tokens and and maintain a large context window. CoT generates a bunch of tokens that the user won’t see and typical system prompting seems to get this done effectively. You may use CoT for scorers in your evaluations.
4. What is KV caching? Why is it important?
    1. See KV caching
5. How do I optimize my prompt for KV caching?
    1. Place dynamic injections as close to the end of the prompt as possible. The more likely the injection is likely to change or get injected in a typical session, the lower the line(s) should go. For example, timestamp should always go at the very bottom of the prompt, and the image generation injection should go at the bottom, but should be high up because it less likely to change every turn.

## Glossary

* Backstory: Set of policies that the AI is required to know about itself. For example, if a user asks the AI it's name and who made it, it can reference its backstory to know it's called Copilot and is made by Microsoft.
* Block: A section of a prompt with it’s own liquid file. This file can be referred to within other prompts to maintain standard sections.
* Dynamic injections: Fields or variables that alter the core system prompt. For example, timestamps, location, and user name are dynamic injections because they change depending on the user and the time of day. We may also inject some extra lines when a user is on a specific device, has a paid subscription, or triggered a tool/answer card.
* Feature: User facing experience (Think Deeper, Podcasts, Suggested Replies)
* Feature Flag: If statement that allows us to enable or disable specific functionality in an application without changing the production experience. This makes it easier to test new features, roll out updates gradually, and quickly respond to issues by turning features off if needed.
* KV caching: Key-Value caching is a performance optimization used in transformer-based language models. Instead of recalculating the keys and values for every token each time a new word is generated, KV caching stores them. This way, the model can reuse the cached data and only compute attention for the new token, dramatically reducing redundant computation.
* Liquid File: Primary file type for system prompts to allow for dynamically injected instructions. It combines static HTML with dynamic placeholders and uses special syntax like {{ variable }} and {% tag %} to insert data, apply logic, or loop through content.
* Picasso: Repository responsible for storing historical contexts, orchestration of models, system prompts etc.
* Prompt: Most likely refering to a “System Prompt.” This is a set of instructions injected every turn, hidden from the user, that guides the model on what tools to invoke or how to respond.
* Tool: External function or API call that the model can invoke to extend its capabilities, such as image generation, file analysis, or web search, that the model itself can't perform natively.
