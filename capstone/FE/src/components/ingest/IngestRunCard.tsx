"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { Spinner } from "@/components/ui/spinner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  Copy,
  Check,
  ExternalLink,
  RefreshCw,
} from "lucide-react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"
const PREFECT_UI = process.env.NEXT_PUBLIC_PREFECT_UI_URL ?? ""
const POLL_INTERVAL_MS = 4_000

// ── Types ───────────────────────────────────────────────────────
export type RunPhase =
  | "accepted"
  | "waiting"
  | "running"
  | "completed"
  | "partial"
  | "failed"

const TERMINAL: ReadonlySet<string> = new Set(["completed", "partial", "failed"])

interface ReportPayload {
  status?: string
  sources_count?: number
  chunks_count?: number
  duration_seconds?: number
  started_at?: string
  finished_at?: string
  errors?: string[]
}

export interface ActiveRun {
  courseName: string
  flowRunId: string
  counts: { slides: number; textbooks: number; videos: number }
}

interface Props {
  run: ActiveRun
}

// ── Helpers ─────────────────────────────────────────────────────
function sanitizeError(raw: string): string {
  // Strip anything that looks like a URL with credentials, a stack trace, or a path
  if (/(?:https?:\/\/|s3:\/\/|\/home\/|\/root\/|Traceback|File ")/.test(raw)) {
    return "Internal processing error"
  }
  return raw.length > 200 ? raw.slice(0, 200) + "…" : raw
}

function fmtDuration(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

function phaseLabel(p: RunPhase): string {
  const map: Record<RunPhase, string> = {
    accepted: "Đã tiếp nhận",
    waiting: "Đang khởi tạo…",
    running: "Đang xử lý…",
    completed: "Hoàn tất",
    partial: "Hoàn tất một phần",
    failed: "Thất bại",
  }
  return map[p]
}

function phaseBadgeVariant(p: RunPhase) {
  if (p === "completed") return "default" as const
  if (p === "partial") return "outline" as const
  if (p === "failed") return "destructive" as const
  return "secondary" as const
}

function PhaseIcon({ phase }: { phase: RunPhase }) {
  switch (phase) {
    case "accepted":
    case "waiting":
    case "running":
      return <Spinner size="sm" />
    case "completed":
      return <CheckCircle className="size-4" style={{ color: "var(--success)" }} />
    case "partial":
      return <AlertTriangle className="size-4" style={{ color: "var(--warning)" }} />
    case "failed":
      return <XCircle className="size-4" style={{ color: "var(--error)" }} />
  }
}

// ── Component ───────────────────────────────────────────────────
export function IngestRunCard({ run }: Props) {
  const { apiFetch } = useAuth()
  const [phase, setPhase] = useState<RunPhase>("accepted")
  const [report, setReport] = useState<ReportPayload | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [pollError, setPollError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Ref to track staleness across closures
  const cancelledRef = useRef(false)

  const poll = useCallback(async () => {
    try {
      const res = await apiFetch(
        `${API}/system/v0/knowledge/ingest-report?course=${encodeURIComponent(run.courseName)}&run_id=${encodeURIComponent(run.flowRunId)}`
      )

      if (cancelledRef.current) return

      if (res.status === 404) {
        // Run hasn't registered in Prefect yet
        setPhase((prev) => (prev === "accepted" ? "waiting" : prev))
        setPollError(null)
        return
      }
      if (!res.ok) {
        setPollError(`Server error (${res.status})`)
        return
      }

      const data: ReportPayload = await res.json()
      if (cancelledRef.current) return

      setReport(data)
      setLastUpdate(new Date())
      setPollError(null)

      const status = data.status?.toUpperCase()
      if (status === "COMPLETED") setPhase("completed")
      else if (status === "PARTIAL") setPhase("partial")
      else if (status === "FAILED") setPhase("failed")
      else if (status === "RUNNING") setPhase("running")
      else setPhase("waiting")
    } catch {
      if (!cancelledRef.current) {
        setPollError("Network error — retrying…")
      }
    }
  }, [apiFetch, run.courseName, run.flowRunId])

  useEffect(() => {
    cancelledRef.current = false
    // Reset state for a new run
    setPhase("accepted")
    setReport(null)
    setPollError(null)

    // First poll immediately
    poll()

    const id = setInterval(() => {
      if (!cancelledRef.current && !TERMINAL.has(phase)) {
        poll()
      }
    }, POLL_INTERVAL_MS)

    return () => {
      cancelledRef.current = true
      clearInterval(id)
    }
    // ponytail: intentionally excluding `phase` from deps — the interval
    // reads it via the closure that `poll` closes over `setPhase`.
    // Including it would restart the interval on every phase change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.flowRunId, poll])

  // Stop polling once terminal
  useEffect(() => {
    if (TERMINAL.has(phase)) {
      cancelledRef.current = true
    }
  }, [phase])

  function handleCopy() {
    navigator.clipboard.writeText(run.flowRunId).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div
      className="rounded-lg border p-4 space-y-3 msg-enter"
      style={{
        borderColor: "var(--se-border)",
        backgroundColor: "var(--surface)",
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold truncate" style={{ color: "var(--ink)" }}>
          {run.courseName}
        </h4>
        <Badge variant={phaseBadgeVariant(phase)}>
          <PhaseIcon phase={phase} />
          {phaseLabel(phase)}
        </Badge>
      </div>

      {/* Flow ID */}
      <div className="flex items-center gap-1.5 text-xs" style={{ color: "var(--ink-muted)" }}>
        <span className="font-mono truncate select-all">{run.flowRunId}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="shrink-0 rounded p-0.5 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
          aria-label="Copy flow ID"
        >
          {copied ? (
            <Check className="size-3.5" style={{ color: "var(--success)" }} />
          ) : (
            <Copy className="size-3.5" />
          )}
        </button>
        {PREFECT_UI && (
          <a
            href={`${PREFECT_UI}/flow-runs/flow-run/${run.flowRunId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 rounded p-0.5 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
            aria-label="Open in Prefect UI"
          >
            <ExternalLink className="size-3.5" />
          </a>
        )}
      </div>

      {/* File counts */}
      <div className="flex gap-4 text-xs" style={{ color: "var(--ink-muted)" }}>
        <span>Slides: {run.counts.slides}</span>
        <span>Giáo trình: {run.counts.textbooks}</span>
        <span>Video: {run.counts.videos}</span>
      </div>

      {/* Report details (once available) */}
      {report && (
        <div
          className="rounded-md p-3 space-y-1.5 text-xs"
          style={{ backgroundColor: "var(--surface-2)", color: "var(--ink)" }}
        >
          {report.sources_count != null && (
            <p>Nguồn đã xử lý: <strong>{report.sources_count}</strong></p>
          )}
          {report.chunks_count != null && (
            <p>Chunks: <strong>{report.chunks_count}</strong></p>
          )}
          {report.duration_seconds != null && (
            <p>Thời lượng: <strong>{fmtDuration(report.duration_seconds)}</strong></p>
          )}
          {report.started_at && (
            <p>Bắt đầu: {new Date(report.started_at).toLocaleTimeString()}</p>
          )}
          {report.finished_at && (
            <p>Kết thúc: {new Date(report.finished_at).toLocaleTimeString()}</p>
          )}
          {report.errors && report.errors.length > 0 && (
            <ul className="mt-1 space-y-0.5" style={{ color: "var(--error)" }}>
              {report.errors.map((e, i) => (
                <li key={i}>⚠ {sanitizeError(e)}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Poll error + retry */}
      {pollError && (
        <div className="flex items-center gap-2 text-xs" style={{ color: "var(--warning)" }}>
          <span>{pollError}</span>
          {TERMINAL.has(phase) || (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs"
              onClick={() => {
                setPollError(null)
                poll()
              }}
            >
              <RefreshCw className="size-3 mr-1" />
              Thử lại
            </Button>
          )}
        </div>
      )}

      {/* Timestamp */}
      <p className="text-[10px]" style={{ color: "var(--ink-muted)" }}>
        Cập nhật: {lastUpdate.toLocaleTimeString()}
      </p>
    </div>
  )
}
