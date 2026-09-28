"use client"

import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"
import { v7 as uuid } from "uuid"
import { useAuth } from "@/contexts/AuthContext"
import { readSession } from "@/lib/study"
import { normaliseUiAction, type Citation, type UiAction } from "@/lib/normalise"
import { useChatPoll, type PollResult } from "@/hooks/useChatPoll"
import { MessageList } from "./MessageList"
import { MessageInput } from "./MessageInput"
import { type Message } from "./MessageBubble"

interface ChatPanelProps {
  sessionId: string | null
  sessionReady: boolean
  pdfOpen: boolean
  onUiAction: (action: UiAction) => void
  onOpenCitation: (citation: Citation) => void
  onSessionActivity: () => void
}

export function ChatPanel({ sessionId, sessionReady, pdfOpen, onUiAction, onOpenCitation, onSessionActivity }: ChatPanelProps) {
  const { apiFetch } = useAuth()
  const [messages, setMessages] = useState<Message[]>([])
  const [loaded, setLoaded] = useState(false)
  const handleDone = useCallback((result: PollResult) => {
    setMessages((previous) => [...previous, {
      id: uuid(), role: "ta", content: result.message, uiAction: result.uiAction,
    }])
    if (result.uiAction && "action" in result.uiAction) onUiAction(result.uiAction)
    onSessionActivity()
  }, [onUiAction, onSessionActivity])
  const { state, error, thought, partial, submit, resumeTask } = useChatPoll(handleDone)

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    readSession(apiFetch, sessionId)
      .then((data) => {
        if (cancelled) return
        setMessages(data.messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          uiAction: normaliseUiAction(message.ui_action),
        })))
        setLoaded(true)
        if (data.pending_chat_id) void resumeTask(data.pending_chat_id)
      })
      .catch(() => {
        if (!cancelled) {
          setLoaded(true)
          toast.error("Không thể tải lịch sử hội thoại.")
        }
      })
    return () => { cancelled = true }
  }, [apiFetch, sessionId, resumeTask])

  useEffect(() => {
    if (state === "fail" && error) toast.error("Trợ lý gặp lỗi", { description: error })
  }, [state, error])

  const handleSubmit = useCallback(async (userInput: string) => {
    if (!sessionId || !sessionReady || !loaded) return
    setMessages((previous) => [...previous, { id: uuid(), role: "user", content: userInput }])
    onSessionActivity()
    await submit(userInput)
  }, [sessionId, sessionReady, loaded, submit, onSessionActivity])

  const handleNavigate = useCallback((course: string, topic: string, page: number) => {
    onUiAction({ action: "NAVIGATE_PDF", course, topic, destination: "", page })
  }, [onUiAction])

  return (
    <div className="flex h-full flex-col" style={{ backgroundColor: "var(--bg)" }}>
      {!loaded && sessionId ? (
        <p className="p-4 text-sm" style={{ color: "var(--ink-muted)" }}>Đang tải hội thoại…</p>
      ) : (
        <MessageList messages={messages} pollState={state} thought={thought} partial={partial}
          error={error} pdfOpen={pdfOpen} onNavigate={handleNavigate} onOpenCitation={onOpenCitation} />
      )}
      <MessageInput onSubmit={handleSubmit} pollState={state} disabled={!sessionId || !sessionReady || !loaded}
        disabledReason={sessionReady && !sessionId ? "Không thể mở phiên. Hãy chọn Chat mới." : "Đang khôi phục phiên…"} />
    </div>
  )
}
