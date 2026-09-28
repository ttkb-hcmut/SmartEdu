"use client"

import { useCallback, useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { AppShell } from "@/components/layout/AppShell"
import { Sidebar } from "@/components/layout/Sidebar"
import { TopBar } from "@/components/layout/TopBar"
import { ChatPanel } from "@/components/chat/ChatPanel"
import { PDFSkeleton } from "@/components/pdf/PDFSkeleton"
import { useAuth } from "@/contexts/AuthContext"
import { parseDocumentTarget, type Citation, type UiAction } from "@/lib/normalise"
import { listSessions, newSession, resumeSession, type SessionSummary } from "@/lib/study"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"
const PDFViewer = dynamic(() => import("@/components/pdf/PDFViewer").then((m) => m.PDFViewer),
  { ssr: false, loading: () => <PDFSkeleton /> })

export default function ChatPage() {
  const router = useRouter()
  const { apiFetch, sessionId, sessionReady, setSessionId } = useAuth()
  const [courses, setCourses] = useState<string[]>([])
  const [topics, setTopics] = useState<string[]>([])
  const [activeCourse, setActiveCourse] = useState<string | null>(null)
  const [openCourse, setOpenCourse] = useState<string | null>(null)
  const [openTopic, setOpenTopic] = useState<string | null>(null)
  const [rawFile, setRawFile] = useState<string | null>(null)
  const [pdfPage, setPdfPage] = useState(1)
  const [mobileTab, setMobileTab] = useState<"chat" | "pdf">("chat")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionsTotal, setSessionsTotal] = useState(0)
  const [historyQuery, setHistoryQuery] = useState("")

  useEffect(() => {
    apiFetch(`${API}/system/v0/knowledge/courses`)
      .then((response) => response.json())
      .then((data) => setCourses(data.courses ?? []))
      .catch(() => toast.error("Không thể tải danh sách môn học."))
  }, [apiFetch])

  useEffect(() => {
    if (!activeCourse) return
    let cancelled = false
    apiFetch(`${API}/system/v0/knowledge/courses/${encodeURIComponent(activeCourse)}/topics`)
      .then((response) => response.json())
      .then((data) => { if (!cancelled) setTopics(data.topics ?? []) })
      .catch(() => { if (!cancelled) setTopics([]) })
    return () => { cancelled = true }
  }, [activeCourse, apiFetch])

  const refreshSessions = useCallback(() => {
    if (!sessionReady) return
    void listSessions(apiFetch, historyQuery).then((data) => {
      setSessions(data.items)
      setSessionsTotal(data.total)
    }).catch(() => {})
  }, [apiFetch, historyQuery, sessionReady])

  useEffect(() => { refreshSessions() }, [refreshSessions, sessionId])

  const openDocument = useCallback((document: string | null, page: number | null) => {
    const target = parseDocumentTarget(document, page)
    if (!target) return
    setActiveCourse(target.course)
    setOpenCourse(target.course)
    setOpenTopic(target.topic)
    setRawFile(target.rawFile)
    setPdfPage(target.page)
    setMobileTab("pdf")
    setSidebarOpen(false)
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.has("document")) queueMicrotask(() => openDocument(params.get("document"), Number(params.get("page"))))
  }, [openDocument])

  const handleUiAction = useCallback((action: UiAction) => {
    setActiveCourse(action.course)
    setOpenCourse(action.course)
    setOpenTopic(action.topic)
    setRawFile(null)
    setPdfPage(action.page)
    setMobileTab("pdf")
  }, [])

  const handleOpenCitation = useCallback((citation: Citation) => {
    openDocument(citation.document, citation.page)
  }, [openDocument])

  const selectSession = useCallback(async (id: string) => {
    try {
      await resumeSession(apiFetch, id)
      setSessionId(id)
      window.localStorage.setItem("smartedu_session", id)
      router.push(`/chat?session=${encodeURIComponent(id)}`)
      setMobileTab("chat")
      setSidebarOpen(false)
    } catch {
      toast.error("Không thể mở lại hội thoại.")
    }
  }, [apiFetch, router, setSessionId])

  const createSession = useCallback(async () => {
    try {
      const { session_id: id } = await newSession(apiFetch)
      setSessionId(id)
      window.localStorage.setItem("smartedu_session", id)
      router.push(`/chat?session=${encodeURIComponent(id)}`)
      setMobileTab("chat")
      setSidebarOpen(false)
      refreshSessions()
    } catch {
      toast.error("Không thể tạo hội thoại mới.")
    }
  }, [apiFetch, router, setSessionId, refreshSessions])

  const loadMoreSessions = useCallback(() => {
    void listSessions(apiFetch, historyQuery, sessions.length).then((data) => {
      setSessions((previous) => [...previous, ...data.items])
      setSessionsTotal(data.total)
    }).catch(() => toast.error("Không thể tải thêm hội thoại."))
  }, [apiFetch, historyQuery, sessions.length])

  const pdfOpen = !!(openCourse && (openTopic || rawFile))
  const breadcrumb = ["Trò chuyện", ...(openCourse ? [openCourse] : []),
    ...(openTopic ? [openTopic, "page.pdf"] : rawFile ? [rawFile] : []), ...(pdfOpen ? [`trang ${pdfPage}`] : [])]
  const topBar = <TopBar breadcrumbs={breadcrumb} onMenu={() => setSidebarOpen(true)}
    mobileTab={mobileTab} pdfAvailable={pdfOpen} onMobileTabChange={setMobileTab} />

  return (
    <AppShell sidebarOpen={sidebarOpen} onCloseSidebar={() => setSidebarOpen(false)} mobileTab={mobileTab}
      sidebar={<Sidebar courses={courses} activeCourse={activeCourse} topics={topics} sessions={sessions}
        activeSessionId={sessionId} historyQuery={historyQuery} hasMoreSessions={sessions.length < sessionsTotal}
        onCourseSelect={(course) => { setActiveCourse(course); setTopics([]) }}
        onTopicSelect={(topic) => { if (activeCourse) openDocument(`${activeCourse}/${topic}/page.pdf`, 1) }}
        onSessionSelect={selectSession} onNewSession={createSession} onHistoryQuery={setHistoryQuery}
        onLoadMoreSessions={loadMoreSessions} />}
      panel={pdfOpen ? <>
        <div className="lg:hidden">{topBar}</div>
        <PDFViewer key={rawFile ? `${openCourse}/_raw/${rawFile}` : `${openCourse}/${openTopic}`}
          course={openCourse!} topic={openTopic ?? ""} rawFile={rawFile ?? undefined}
          page={pdfPage} onPageChange={setPdfPage} className="min-h-0 flex-1" />
      </> : undefined}>
      {topBar}
      <ChatPanel key={sessionId ?? "pending"} sessionId={sessionId} sessionReady={sessionReady}
        pdfOpen={pdfOpen} onUiAction={handleUiAction} onOpenCitation={handleOpenCitation}
        onSessionActivity={refreshSessions} />
    </AppShell>
  )
}
