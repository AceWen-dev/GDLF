# ADR 0009 — Candidates and claims are owner-approved registries

## Status

Accepted.

## Context

Candidates (which runs enter the shortlist) and claims (which numbers get
reported) are the owner's research decisions. With agents running experiments
autonomously, nothing stopped an agent from shortlisting a run set, promoting
it, or creating a claim from its own judgment. The chat log was the only place
an approval ever existed, and it is not part of the record.

## Decision

- `Claim.approved_by` and `Candidate.approved_by` are required, non-empty
  fields recording the owner's approval (who, when, and ideally the quote).
- `ClaimStore.add` refuses a claim without `approved_by`.
- `CandidateStore.add` refuses a candidate without `approved_by`, and a record
  cannot change status until it carries one (legacy records must be
  re-approved by the owner first).
- The template tool makes the gate explicit: `add` / `promote` / `reject` /
  `archive` require `--approved-by`, which must quote the owner's answer.
- The agent workflow is fixed: **propose, wait, record**. An agent prepares
  the exact command and the expected record, asks the owner, and only then
  executes with the approval quoted. An agent never writes `--approved-by`
  from its own judgment.
- Removing a claim or candidate is an owner action, not an agent action.
- The report and `INDEX.md` render the approver; records without one show
  `UNAPPROVED`, and INDEX warns when any exist.

## Consequences

- Every entry in the shortlist and every reported number says who signed it
  off — the audit trail lives in the record, not in a chat log.
- Agents are deliberately slowed down at the two highest-leverage decision
  points; triage and reporting stay human-paced.
- A dishonest agent could fabricate an approval string; the countermeasure is
  review — approvers are visible in INDEX/report, and the owner can grep for
  approvals they never gave.
- Legacy records created before this ADR load with an empty `approved_by`,
  cannot change status, and are listed as `UNAPPROVED` until the owner
  re-approves them.
