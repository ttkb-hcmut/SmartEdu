"use client"

import { useAuth } from "@/contexts/AuthContext"
import { Settings, ArrowLeft, Menu } from "lucide-react"
import Link from "next/link"

interface TopBarProps {
  title?: string
  backHref?: string
  breadcrumbs?: string[]
  onMenu?: () => void
  mobileTab?: "chat" | "pdf"
  pdfAvailable?: boolean
  onMobileTabChange?: (tab: "chat" | "pdf") => void
}

export function TopBar({ title, backHref, breadcrumbs, onMenu, mobileTab, pdfAvailable, onMobileTabChange }: TopBarProps) {
  const { sessionId } = useAuth()

  return (
    <header
      className="flex h-12 shrink-0 items-center gap-3 border-b px-4"
      style={{
        backgroundColor: "var(--bg)",
        borderColor: "var(--se-border)",
      }}
    >
      {onMenu && <button type="button" aria-label="Mở menu" onClick={onMenu} className="rounded p-1 lg:hidden"><Menu className="size-5" /></button>}
      {backHref && (
        <Link
          href={backHref}
          aria-label="Quay lại"
          className="flex size-7 items-center justify-center rounded-xs transition-colors hover:bg-[var(--se-accent-subtle)]"
          style={{ color: "var(--ink-muted)" }}
        >
          <ArrowLeft className="size-4" />
        </Link>
      )}

      {breadcrumbs?.length ? (
        <nav aria-label="Đường dẫn" className="min-w-0 truncate text-sm" style={{ color: "var(--ink-muted)" }}>
          {breadcrumbs.map((part, index) => <span key={`${part}-${index}`}><span className={index === breadcrumbs.length - 1 ? "font-medium" : ""}>{part}</span>{index < breadcrumbs.length - 1 && <span className="px-1">›</span>}</span>)}
        </nav>
      ) : title && (
        <span
          className="text-sm font-medium"
          style={{ color: "var(--ink)" }}
        >
          {title}
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        {onMobileTabChange && <div className="flex gap-1 rounded border p-0.5 lg:hidden" style={{ borderColor: "var(--se-border)" }}>
          <button type="button" onClick={() => onMobileTabChange("chat")} aria-pressed={mobileTab === "chat"}
            className="rounded px-2 py-1 text-xs aria-pressed:bg-[var(--se-accent-subtle)]">Chat</button>
          <button type="button" onClick={() => onMobileTabChange("pdf")} disabled={!pdfAvailable} aria-pressed={mobileTab === "pdf"}
            className="rounded px-2 py-1 text-xs aria-pressed:bg-[var(--se-accent-subtle)] disabled:opacity-40">PDF</button>
        </div>}
        {sessionId && (
          <span
            className="hidden items-center gap-1.5 text-xs md:flex"
            style={{ color: "var(--ink-muted)" }}
          >
            <span
              className="size-1.5 rounded-full"
              style={{ backgroundColor: "var(--success)" }}
            />
            Phiên đang hoạt động
          </span>
        )}

        <Link
          href="/settings"
          aria-label="Cài đặt"
          className="flex size-7 items-center justify-center rounded-xs transition-colors hover:bg-[var(--se-accent-subtle)]"
          style={{ color: "var(--ink-muted)" }}
        >
          <Settings className="size-4" />
        </Link>
      </div>
    </header>
  )
}
