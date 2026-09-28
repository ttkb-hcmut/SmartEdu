"use client"

import ReactMarkdown from "react-markdown"
import { type Citation, type UiActionPayload } from "@/lib/normalise"
import { SlideChip } from "./SlideChip"
import { cn } from "@/lib/utils"

export interface Message {
  id: string
  role: "user" | "ta"
  content: string
  uiAction?: UiActionPayload | null
}

interface MessageBubbleProps {
  message: Message
  pdfOpen: boolean
  onNavigate: (course: string, topic: string, page: number) => void
  onOpenCitation: (citation: Citation) => void
}

export function MessageBubble({ message, pdfOpen, onNavigate, onOpenCitation }: MessageBubbleProps) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "msg-enter flex flex-col gap-1",
        isUser ? "items-end" : "items-start"
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-sm px-3.5 py-2.5 text-sm leading-relaxed",
          isUser
            ? "rounded-br-xs"
            : "rounded-bl-xs"
        )}
        style={{
          backgroundColor: isUser ? "var(--surface-2)" : "var(--surface)",
          color: "var(--ink)",
          border: "1px solid var(--se-border)",
        }}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none [&_a]:text-[var(--se-accent)] [&_code]:rounded-xs [&_code]:bg-[var(--surface-2)] [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-xs [&_pre]:rounded-sm [&_pre]:bg-[var(--surface-2)] [&_pre]:p-3 [&_strong]:font-semibold">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>

      {/* Slide navigation chip — only shown when viewer is open OR just opened */}
      {!isUser && message.uiAction && "citations" in message.uiAction && (
        <div className="flex max-w-[80%] flex-wrap gap-2" aria-label="Sources">
          {message.uiAction.citations.map((citation) => (
            <button
              key={citation.uri}
              type="button"
              disabled={!citation.document || !citation.page}
              onClick={() => onOpenCitation(citation)}
              className="rounded border px-2 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50"
              title={citation.document ?? "PDF unavailable"}
            >
              {citation.document
                ? citation.document.endsWith("/page.pdf")
                  ? citation.document.split("/").at(-2)
                  : citation.document.split("/").at(-1)
                : citation.uri}
              {citation.page ? ` · p. ${citation.page}` : " · source only"}
            </button>
          ))}
        </div>
      )}
      {!isUser && message.uiAction && "action" in message.uiAction && (
        <SlideChip
          page={message.uiAction.page}
          visible={pdfOpen}
          onClick={() =>
            onNavigate(
              (message.uiAction as Extract<UiActionPayload, {action: "NAVIGATE_PDF"}>).course,
              (message.uiAction as Extract<UiActionPayload, {action: "NAVIGATE_PDF"}>).topic,
              (message.uiAction as Extract<UiActionPayload, {action: "NAVIGATE_PDF"}>).page
            )
          }
        />
      )}
    </div>
  )
}
