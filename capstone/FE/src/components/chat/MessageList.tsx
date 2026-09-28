"use client"

import { useEffect, useRef } from "react"
import { MessageBubble, type Message } from "./MessageBubble"
import { ThoughtIndicator } from "./ThoughtIndicator"
import { type PollState, type AgentThought } from "@/hooks/useChatPoll"
import { type Citation, type UiAction } from "@/lib/normalise"
import { cn } from "@/lib/utils"

interface MessageListProps {
  messages: Message[]
  pollState: PollState
  thought: AgentThought | null
  partial: string
  error: string | null
  pdfOpen: boolean
  onNavigate: (course: string, topic: string, page: number) => void
  onOpenCitation: (citation: Citation) => void
  className?: string
}

export function MessageList({
  messages,
  pollState,
  thought,
  partial,
  error,
  pdfOpen,
  onNavigate,
  onOpenCitation,
  className,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, pollState])

  return (
    <div
      className={cn(
        "flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-4",
        className
      )}
    >
      {messages.length === 0 && (
        <div
          className="flex flex-1 flex-col items-center justify-center gap-2 text-center"
          style={{ color: "var(--ink-muted)" }}
        >
          <p className="text-sm">Hãy đặt câu hỏi cho trợ lý học tập của bạn.</p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          pdfOpen={pdfOpen}
          onNavigate={onNavigate}
          onOpenCitation={onOpenCitation}
        />
      ))}

      {pollState === "polling" &&
        (partial ? (
          <MessageBubble
            message={{ id: "__streaming__", role: "ta", content: partial }}
            pdfOpen={pdfOpen}
            onNavigate={onNavigate}
            onOpenCitation={onOpenCitation}
          />
        ) : (
          <ThoughtIndicator thought={thought} />
        ))}

      {(pollState === "fail" || pollState === "timeout") && error && (
        <p
          className="rounded-xs border px-3 py-2 text-sm"
          style={{
            color: "var(--error)",
            borderColor: "color-mix(in oklch, var(--error) 30%, transparent)",
            backgroundColor: "color-mix(in oklch, var(--error) 8%, transparent)",
          }}
          role="alert"
        >
          {pollState === "timeout"
            ? "Trợ lý không phản hồi sau 5 phút. Vui lòng thử lại."
            : error}
        </p>
      )}

      <div ref={bottomRef} className="h-px" aria-hidden="true" />
    </div>
  )
}

export type { Message, UiAction }
