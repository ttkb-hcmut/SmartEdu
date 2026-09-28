"use client"

import Link from "next/link"
import { BookOpen, FileText, History, LogOut, Plus, Search, Settings, Upload, Waypoints } from "lucide-react"
import { useAuth } from "@/contexts/AuthContext"
import { type SessionSummary } from "@/lib/study"
import { cn } from "@/lib/utils"

interface SidebarProps {
  courses: string[]
  activeCourse: string | null
  topics: string[]
  sessions: SessionSummary[]
  activeSessionId: string | null
  historyQuery: string
  hasMoreSessions: boolean
  onCourseSelect: (course: string) => void
  onTopicSelect: (topic: string) => void
  onSessionSelect: (id: string) => void
  onNewSession: () => void
  onHistoryQuery: (query: string) => void
  onLoadMoreSessions: () => void
}

export function Sidebar(props: SidebarProps) {
  const { logout, isAdmin, sessionReady } = useAuth()
  const {
    courses, activeCourse, topics, sessions, activeSessionId, historyQuery, hasMoreSessions,
    onCourseSelect, onTopicSelect, onSessionSelect, onNewSession, onHistoryQuery, onLoadMoreSessions,
  } = props

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-14 shrink-0 items-center border-b px-4" style={{ borderColor: "var(--se-border)" }}>
        <Link href="/chat" className="text-sm font-semibold tracking-tight" style={{ color: "var(--se-primary)" }}>SmartEdu</Link>
      </div>

      <nav className="flex-1 space-y-5 overflow-y-auto p-3" aria-label="Điều hướng học tập">
        <div className="space-y-1">
          <button type="button" onClick={onNewSession} disabled={!sessionReady}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-2 text-sm font-medium hover:bg-[var(--se-accent-subtle)] disabled:opacity-50">
            <Plus className="size-4" /> Chat mới
          </button>
          <Link href="/search" className="flex items-center gap-2 rounded-sm px-2 py-2 text-sm hover:bg-[var(--se-accent-subtle)]"><Search className="size-4" /> Tìm kiếm</Link>
          <Link href="/roadmap" className="flex items-center gap-2 rounded-sm px-2 py-2 text-sm hover:bg-[var(--se-accent-subtle)]"><Waypoints className="size-4" /> Roadmap & tiến độ</Link>
          {isAdmin && <Link href="/admin/ingest" className="flex items-center gap-2 rounded-sm px-2 py-2 text-sm hover:bg-[var(--se-accent-subtle)]"><Upload className="size-4" /> Nạp tài liệu</Link>}
        </div>

        <section aria-label="Môn học">
          <h2 className="mb-1 px-2 text-[11px] font-medium uppercase tracking-widest" style={{ color: "var(--ink-muted)" }}>Môn học</h2>
          {courses.length === 0 && <p className="px-2 text-xs" style={{ color: "var(--ink-muted)" }}>Chưa có môn học nào.</p>}
          {courses.map((course) => (
            <div key={course}>
              <button type="button" onClick={() => onCourseSelect(course)}
                className={cn("flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-[var(--se-accent-subtle)]", activeCourse === course && "font-medium")}
                aria-expanded={activeCourse === course}>
                <BookOpen className="size-3.5 shrink-0" /><span className="truncate">{course}</span>
              </button>
              {activeCourse === course && topics.map((topic) => (
                <button key={topic} type="button" onClick={() => onTopicSelect(topic)}
                  className="ml-5 flex w-[calc(100%-1.25rem)] items-center gap-2 rounded-sm px-2 py-1 text-left text-xs hover:bg-[var(--se-accent-subtle)]">
                  <FileText className="size-3 shrink-0" /><span className="truncate">{topic}</span>
                </button>
              ))}
            </div>
          ))}
        </section>

        <section aria-label="Lịch sử hội thoại">
          <h2 className="mb-2 flex items-center gap-2 px-2 text-[11px] font-medium uppercase tracking-widest" style={{ color: "var(--ink-muted)" }}><History className="size-3" /> Hội thoại</h2>
          <input value={historyQuery} onChange={(event) => onHistoryQuery(event.target.value)}
            placeholder="Lọc lịch sử…" aria-label="Lọc lịch sử hội thoại"
            className="mb-2 w-full rounded-sm border bg-transparent px-2 py-1.5 text-xs" style={{ borderColor: "var(--se-border)" }} />
          {sessions.map((session) => (
            <button key={session.id} type="button" onClick={() => onSessionSelect(session.id)}
              aria-current={activeSessionId === session.id ? "page" : undefined}
              className={cn("block w-full truncate rounded-sm px-2 py-1.5 text-left text-sm hover:bg-[var(--se-accent-subtle)]", activeSessionId === session.id && "bg-[var(--se-accent-subtle)] font-medium")}
              title={session.name}>{session.name}</button>
          ))}
          {sessions.length === 0 && <p className="px-2 text-xs" style={{ color: "var(--ink-muted)" }}>Không có hội thoại.</p>}
          {hasMoreSessions && <button type="button" onClick={onLoadMoreSessions} className="px-2 py-1 text-xs underline">Xem thêm</button>}
        </section>
      </nav>

      <div className="shrink-0 space-y-1 border-t p-2" style={{ borderColor: "var(--se-border)" }}>
        <Link href="/settings" className="flex items-center gap-2 rounded-sm px-3 py-1.5 text-sm hover:bg-[var(--se-accent-subtle)]"><Settings className="size-3.5" /> Cài đặt</Link>
        <button type="button" onClick={() => logout()} className="flex w-full items-center gap-2 rounded-sm px-3 py-1.5 text-left text-sm hover:bg-[var(--se-accent-subtle)]"><LogOut className="size-3.5" /> Đăng xuất</button>
      </div>
    </div>
  )
}
