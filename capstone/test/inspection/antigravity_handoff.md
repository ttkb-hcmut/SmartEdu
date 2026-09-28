# Antigravity handoff — ingestion observability surface

## Mission

Make the existing browser ingestion flow observable for the MuSiQue inspection run.
The operator must see the accepted flow identity and the terminal ingestion report without opening server logs.

This is an observability/UI task. Do not change ingestion behavior, Prefect orchestration, repositories, benchmark scoring, or database schemas.

## Workspace and ownership

Primary workspace:

```text
D:\Project\AI\Capstone
```

Owned implementation area:

```text
capstone/FE/src/components/ingest/
```

Small supporting frontend API/type changes are allowed only under:

```text
capstone/FE/src/
```

Do not edit Python files, Prefect files, Docker files, benchmark files, or database code.
Do not commit. Return a diff summary and manual verification steps.

## Existing backend contract

The ingestion request is:

```text
POST /system/v0/knowledge/ingest-course
```

Successful response is HTTP 202 and contains:

```json
{
  "status": "accepted",
  "flow_run_id": "<prefect-flow-run-id>",
  "details": {
    "slides_count": 0,
    "textbooks_count": 0,
    "videos_count": 0
  }
}
```

The report endpoint is:

```text
GET /system/v0/knowledge/ingest-report?course=<course>&run_id=<flow_run_id>
```

The report may return 404 briefly while the Prefect run is starting. Treat that as `waiting`, not failure.
Poll until the report status is one of:

```text
COMPLETED, PARTIAL, FAILED
```

The report includes source counts, duration, errors, start/finish timestamps, and terminal status.

## Required behavior

1. Preserve the submitted course name and returned `flow_run_id`.
2. After HTTP 202, show a visible run card containing:
   - course name;
   - flow ID with copy action;
   - submitted file counts;
   - current state: `accepted`, `waiting`, `RUNNING`, `COMPLETED`, `PARTIAL`, or `FAILED`;
   - last update time.
3. Poll the report endpoint using the exact course and flow ID.
4. Render terminal counts, duration, and sanitized errors.
5. Keep the run card visible after the form resets.
6. Stop polling on terminal state, component unmount, or a new submission.
7. Handle network errors with a retry action and preserve the flow ID.
8. Never expose access tokens, signed MinIO URLs, passwords, or raw stack traces.
9. If `NEXT_PUBLIC_PREFECT_UI_URL` exists, provide a link to the Prefect flow run. Otherwise do not render a broken link.
10. Keep existing upload progress and validation behavior intact.

## Suggested state model

Use one explicit state object rather than scattered booleans:

```text
idle
uploading
accepted
waiting
running
completed
partial
failed
```

Do not infer terminal state from an HTTP 202 response. The 202 only means that the run was accepted.

## Manual acceptance procedure

1. Start the backend and frontend using the repository’s documented commands.
2. Open the admin ingestion page.
3. Select a MuSiQue-derived textbook/slide/audio bundle and submit it.
4. Capture the 202 response in browser DevTools and confirm the flow ID shown in the UI matches it.
5. Confirm the UI visibly transitions through startup/waiting and running states.
6. Open the Prefect UI and compare the displayed flow ID and terminal state.
7. Compare the UI report with `GET /ingest-report?...`.
8. Capture screenshots named:

```text
ingest-01-upload-progress.png
ingest-02-accepted-flow-id.png
ingest-03-running-report.png
ingest-04-terminal-report.png
```

9. Repeat once with a failed ingestion and confirm `FAILED` or `PARTIAL` is visible with a safe error message.
10. Report any mismatch as a defect with the flow ID, timestamp, screenshot, and request URL.

## Quality constraints

- Prefer existing UI primitives and hooks.
- Keep polling cancellation and stale-response handling explicit.
- Avoid a new dependency.
- Keep the change small and readable.
- Do not add unit-test-only abstractions; manual browser verification is the primary acceptance evidence for this slice.

## Handoff output

Return:

1. changed files;
2. state transition summary;
3. manual verification result;
4. remaining limitations;
5. no commit.
