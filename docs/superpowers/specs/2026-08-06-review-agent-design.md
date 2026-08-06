# Review Agent Design

## Goal

Replace the current merge request reviewer implementation with a provider-neutral `review_agent` that:

- preserves the existing two-phase CLI workflow: `review` then `publish`
- preserves the published GitLab review comment format and behavior
- removes all `Codex` naming and authentication assumptions from the codebase and CI job
- uses an Anthropic-compatible API backend so direct Anthropic and `z.ai` both work through configuration only

## Non-Goals

- changing the GitLab publish schema or inline comment semantics
- introducing multi-provider abstraction beyond the single Anthropic-compatible protocol
- redesigning the review prompt into a materially different review policy
- changing merge request job rules beyond the minimum required rename and variable migration

## Current State

The repository currently contains a review flow built around:

- a review CLI package under `scripts/codex_review`
- a GitLab CI job named `codex-review`
- CI authentication based on `CODEX_AUTH_JSON_B64`
- a local CLI execution model that shells out to `codex exec`

The publish phase already has the right shape and should be preserved as much as possible.

## Target Architecture

The new implementation will live under a neutral package:

- `scripts/review_agent/cli.py`
- `scripts/review_agent/review_runner.py`
- `scripts/review_agent/anthropic_compatible_client.py`
- `scripts/review_agent/publish_runner.py`
- `scripts/review_agent/gitlab_client.py`
- `scripts/review_agent/schema.py`
- `scripts/review_agent/common.py`
- `scripts/review_agent/prompts/review_prompt.md`

The package keeps the existing logical CLI contract:

- `python -m scripts.review_agent.cli review ...`
- `python -m scripts.review_agent.cli publish ...`

The flow remains two-phase:

1. `review` collects diff context, builds the prompt, calls the Anthropic-compatible API, validates structured JSON, and writes the output artifact.
2. `publish` reads the artifact and publishes inline comments and the final summary note to GitLab.

## Naming Constraints

No new code, job names, environment variables, prompts, logs, or user-facing messages should contain `Codex`.

The implementation should also avoid hardcoding `Claude` into the core package naming. The backend protocol is Anthropic-compatible, but the reviewer itself is a neutral `review_agent`.

`anthropic_compatible` is allowed in internal names because it describes the request/response contract rather than branding the feature around a specific vendor.

## CLI Contract

The command shape remains stable:

- `review` generates a structured review artifact
- `publish` posts that artifact to GitLab

The argument model should remain as close as possible to the current one so CI migration is mostly a module-path and variable rename, not a behavioral rewrite.

The module path will change from `scripts.codex_review.cli` to `scripts.review_agent.cli`. This is acceptable because the user explicitly wants all `Codex` references removed.

## Review Phase

The review phase will:

1. resolve `base_sha`
2. collect changed files from local git state
3. load the review prompt from disk
4. append review context that includes:
   - base SHA
   - changed files
   - instruction to limit analysis to those files while using local repository state and git history
5. submit the request to the Anthropic-compatible API
6. parse the model response as JSON
7. validate the response against the structured review schema
8. write the validated artifact to disk

The review phase must fail fast on:

- missing required environment variables
- request transport errors
- non-success API responses
- invalid JSON
- schema validation failures

It must not silently continue into the publish phase after a malformed review result.

## Publish Phase

The publish phase should be migrated with minimal functional change.

It will continue to:

- read the structured review artifact
- post inline discussions where location mapping succeeds
- fall back to a top-level note when inline publication cannot be completed for a finding
- post a success note when there are no findings

The current review result schema should be preserved unless a migration blocker is discovered during implementation.

## Structured Output Contract

The model must return structured JSON that remains compatible with the existing schema used by the current publish flow.

This is the key compatibility boundary of the migration. Prompt changes, API changes, and runner changes are allowed as long as this contract remains stable.

If implementation discovers that the current schema is too coupled to old naming, the schema may be cosmetically renamed internally, but the serialized artifact shape should remain unchanged.

## API Client Design

The API client should be built as an Anthropic-compatible HTTP client rather than a direct dependency on a local CLI.

Responsibilities:

- construct authenticated requests
- send prompts to the configured base URL
- expose the response text payload back to the review runner
- avoid leaking secrets in logs and exceptions

The client should support:

- official Anthropic API when no custom base URL is configured
- `z.ai` when it exposes an Anthropic-compatible endpoint

This is achieved through configuration only, not provider-specific branching logic.

## Environment Variables

Required:

- `ANTHROPIC_API_KEY`
- `GITLAB_API_TOKEN`

Recommended:

- `ANTHROPIC_MODEL`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_API_VERSION`
- `REVIEW_AGENT_OUTPUT_FILE`

Default behavior:

- if `ANTHROPIC_BASE_URL` is unset, use the official Anthropic endpoint
- if `ANTHROPIC_BASE_URL` is set, use it exactly as the API base
- if `ANTHROPIC_API_VERSION` is unset, use a safe default supported by the chosen endpoint
- if `ANTHROPIC_MODEL` is unset in code, CI should provide a default

## GitLab CI Design

The GitLab job should be renamed from `codex-review` to `review-agent`.

The job should:

- keep the current merge-request-only manual trigger behavior
- keep the two-phase `review` then `publish` execution inside one job
- keep artifact retention behavior unless implementation reveals a strong reason to change it
- keep `allow_failure: true` during the migration phase

The job must stop depending on:

- `CODEX_AUTH_JSON_B64`
- `CODEX_HOME`
- `CODEX_SANDBOX_MODE`
- `OPENAI_MODEL`
- the `codex` CLI installation and login check

The job should instead:

- validate `ANTHROPIC_API_KEY` before review starts
- set a default `ANTHROPIC_MODEL`
- run `python -m scripts.review_agent.cli review`
- run `python -m scripts.review_agent.cli publish`

`REVIEW_AGENT_OUTPUT_FILE` should replace `CODEX_REVIEW_OUTPUT_FILE`.

## Prompt Design

The prompt should preserve the strict senior-review posture of the current implementation while removing all `Codex` references.

The prompt must:

- request only structured JSON matching the schema
- focus the review on the changed files
- allow use of local repository state and git history as supporting context
- discourage speculative or low-confidence findings

Recommended prompt rule:

- if confidence is low or the issue is not actionable, omit the finding rather than emit weak review noise

## Error Handling

Failure cases and expected behavior:

- missing `ANTHROPIC_API_KEY`: fail immediately with a clear configuration error
- bad `ANTHROPIC_BASE_URL`: fail with a configuration or transport error that includes the base URL but never the secret
- unsupported model: fail with a clear model/configuration error
- invalid model response JSON: fail the review phase before publish
- valid but empty findings: publish a no-issues-completed note
- inline discussion publish failure for one finding: preserve the current fallback behavior if it already exists in the publish layer

Logs should be explicit enough to diagnose CI failures without printing:

- API keys
- full auth headers
- unnecessary prompt contents if they may contain sensitive code excerpts

## Testing Strategy

Implementation should include tests for:

- prompt assembly from changed file context
- environment-variable validation
- API response parsing
- schema validation on happy-path and malformed payloads
- CI-oriented runner behavior where review failure prevents publish
- publish behavior compatibility with existing no-finding and inline-fallback cases

The migration should also include at least one targeted regression test that proves the artifact generated by `review` is still acceptable to the existing publish flow.

## Migration Strategy

1. add the new neutral package `scripts/review_agent`
2. port shared logic from the current implementation
3. replace CLI-backed review execution with Anthropic-compatible HTTP execution
4. migrate prompt wording
5. migrate tests
6. update the GitLab CI job name, commands, and variables
7. remove the old `scripts/codex_review` package
8. remove any remaining `Codex` references from CI, docs, prompts, and logs

Migration should be completed as one coherent change set rather than leaving both review systems active in parallel. The user explicitly wants no `Codex` references left afterward.

## Open Decisions Already Resolved

- Provider model: Anthropic-compatible HTTP API
- Secret source: GitLab CI Variables
- Required secret: `ANTHROPIC_API_KEY`
- Compatibility target: direct Anthropic and `z.ai` through configuration only
- Reviewer package naming: neutral `review_agent`
- Publish semantics: preserved
- Old `Codex` naming: fully removed

## Success Criteria

The migration is successful when:

- no active reviewer code or CI path depends on `codex` CLI
- no active reviewer code or CI path depends on `CODEX_AUTH_JSON_B64`
- no reviewer-related code, prompt, job name, or env var contains `Codex`
- the review phase runs against an Anthropic-compatible endpoint using `ANTHROPIC_API_KEY`
- `z.ai` can be used by setting `ANTHROPIC_BASE_URL` without code changes
- GitLab review comments are published in the same format as before
- the CLI still exposes `review` and `publish`

## Scope Boundaries

This spec covers the reviewer migration only.

It does not include:

- broader CI cleanup unrelated to the reviewer
- support for non-Anthropic-compatible providers
- generalized provider plugins
- prompt experimentation beyond what is needed to preserve review quality after migration
