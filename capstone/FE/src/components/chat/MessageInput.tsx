"use client"

import { useRef, type KeyboardEvent } from "react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Send } from "lucide-react"
import { cn } from "@/lib/utils"
import { type PollState } from "@/hooks/useChatPoll"

interface MessageInputProps {
  onSubmit: (value: string) => void
  pollState: PollState
  disabled?: boolean
  disabledReason?: string
  className?: string
}

export function MessageInput({ onSubmit, pollState, disabled = false, disabledReason = "Đang khôi phục phiên…", className }: MessageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isPolling = pollState === "polling"

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const value = textareaRef.current?.value.trim()
    if (!value || isPolling || disabled) return
    onSubmit(value)
    if (textareaRef.current) textareaRef.current.value = ""
  }

  return (
    <div
      className={cn("flex items-end gap-2 border-t p-3", className)}
      style={{ borderColor: "var(--se-border)", backgroundColor: "var(--bg)" }}
    >
      <textarea
        ref={textareaRef}
        rows={1}
        placeholder={disabled ? disabledReason : isPolling ? "Đang xử lý…" : "Nhập câu hỏi… (Enter để gửi)"}
        disabled={isPolling || disabled}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex-1 resize-none rounded-xs border px-3 py-2 text-sm",
          "focus:outline-none focus:ring-2 focus:ring-[--ring] focus:ring-offset-1",
          "disabled:cursor-not-allowed disabled:opacity-60",
          "max-h-36 min-h-[38px] overflow-y-auto leading-relaxed"
        )}
        style={{
          backgroundColor: "var(--bg)",
          borderColor: "var(--se-border)",
          color: "var(--ink)",
        }}
        onInput={(e) => {
          // Auto-grow
          const el = e.currentTarget
          el.style.height = "auto"
          el.style.height = `${Math.min(el.scrollHeight, 144)}px`
        }}
      />

      <Button
        variant="accent"
        size="icon"
        onClick={submit}
        disabled={isPolling || disabled}
        aria-label="Gửi tin nhắn"
        className="shrink-0"
      >
        {isPolling ? <Spinner size="sm" /> : <Send className="size-4" />}
      </Button>
    </div>
  )
}
