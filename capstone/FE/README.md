# SmartEdu frontend

Next.js UI for studying with the TA, course PDFs, search, and a read-only roadmap.

## Run

Install dependencies with `npm install`, then run `npm run dev` in this directory. The app listens on port 3002. Set `NEXT_PUBLIC_API_URL` to the backend origin (default `http://localhost:5000`). The backend must expose Student, Knowledge and TA routes under `/system/v0` and the browser must be allowed by its CORS settings.

## Pages

- `/chat`: persistent conversations beside a PDF viewer. On phones, switch between Chat and PDF tabs. The course selection filters the document browser only; TA chat keeps its global retrieval scope.
- `/search`: concept and textbook passage search, optionally filtered by course. Resolved hits open their PDF at the cited page.
- `/roadmap`: course outline, prerequisite connections, and personal mastery when available.
- `/admin/ingest`: course upload for admin accounts.
- `/settings`: language and theme.

Login stores the refresh token in an HTTP-only cookie through the Next API proxy. All backend requests use `apiFetch` for Bearer access tokens and refresh retries. The session ID in browser storage is only a UI preference; the backend verifies session ownership on every read and resume.

## API endpoints used by the study pages

Student: `GET /sessions`, `GET /sessions/{id}`, `POST /sessions/{id}/resume`, `POST /session/start`.

Knowledge: `GET /courses`, `GET /courses/{course}/topics`, `GET /search`, authenticated PDF reads.

TA: `POST /chat`, `GET /chat/status/{task_id}`, `GET /chat/stream/{task_id}`, `GET /roadmap/{course}`.

## Checks

Run `npx tsc --noEmit`, `npm run lint`, and `npm run build` from this directory.
