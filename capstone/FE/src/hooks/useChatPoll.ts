"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { normaliseUiAction, type UiActionPayload } from "@/lib/normalise"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"

export type PollState = "idle" | "polling" | "done" | "fail" | "timeout"

export interface PollResult {
  taskId: string
  message: string
  uiAction: UiActionPayload | null
}

export interface AgentThought {
  agentName?: string
  intent?: string
  thought?: string
}

interface TaskStatusResponse {
  status: string
  error?: string
  result?: { message: string; ui_action?: UiActionPayload | null }
}

export function useChatPoll(onDone?: (result: PollResult) => void) {
  const { apiFetch, sessionId, language } = useAuth()
  const [state, setState] = useState<PollState>("idle")
  const [result, setResult] = useState<PollResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [thought, setThought] = useState<AgentThought | null>(null)
  const [partial, setPartial] = useState<string>("") // live answer text as tokens stream
  const abortRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  const onDoneRef = useRef(onDone)
  const completedTaskIdsRef = useRef(new Set<string>())

  useEffect(() => { onDoneRef.current = onDone }, [onDone])

  const completeTask = useCallback((taskId: string, message: string, uiAction: UiActionPayload | null) => {
    if (completedTaskIdsRef.current.has(taskId)) return
    completedTaskIdsRef.current.add(taskId)
    const completed = { taskId, message, uiAction: normaliseUiAction(uiAction) }
    setResult(completed)
    setThought(null)
    setState("done")
    onDoneRef.current?.(completed)
  }, [])

  const failTask = useCallback((message: string) => {
    setThought(null)
    setState("fail")
    setError(message)
  }, [])

  const recoverTask = useCallback(async (taskId: string, signal: AbortSignal) => {
    setState("polling")
    setError(null)
    setPartial("")
    setThought({ agentName: "TA", thought: "Luồng trả lời bị ngắt; đang lấy trạng thái từ máy chủ…" })
    let consecutiveErrors = 0

    while (mountedRef.current && !signal.aborted) {
      try {
        const response = await apiFetch(`${API}/system/v0/ta/chat/status/${encodeURIComponent(taskId)}`, { signal })
        if (signal.aborted || !mountedRef.current) return
        if (response.status === 404) {
          failTask("Không tìm thấy tác vụ trên máy chủ. Hãy tải lại phiên để kiểm tra lịch sử; câu hỏi không bị gửi lại.")
          return
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`)

        const task = await response.json() as TaskStatusResponse
        if (task.status === "finished" && task.result) {
          completeTask(taskId, task.result.message, task.result.ui_action ?? null)
          return
        }
        if (task.status === "Fail" || task.status === "error") {
          failTask(task.error ?? "Tác vụ trợ lý thất bại.")
          return
        }
        if (task.status !== "working") {
          failTask("Máy chủ trả về trạng thái tác vụ không hợp lệ.")
          return
        }
        consecutiveErrors = 0
      } catch (err) {
        if (signal.aborted || (err as Error)?.name === "AbortError") return
        if (err instanceof Error && err.message.startsWith("Session expired")) {
          failTask("Phiên đăng nhập hết hạn. Hãy đăng nhập lại để kiểm tra câu trả lời đã lưu.")
          return
        }
        consecutiveErrors += 1
        if (consecutiveErrors >= 5) {
          failTask("Mất kết nối khi khôi phục câu trả lời. Hãy tải lại phiên để kiểm tra kết quả; câu hỏi không bị gửi lại.")
          return
        }
      }

      await new Promise((resolve) => setTimeout(resolve, 1500))
    }
  }, [apiFetch, completeTask, failTask])

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false; abortRef.current?.abort() }
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setState("idle")
    setResult(null)
    setError(null)
    setThought(null)
    setPartial("")
  }, [])

  const streamTask = useCallback(
    async (taskId: string) => {
      if (!mountedRef.current) return
      const ctrl = new AbortController()
      abortRef.current = ctrl
      let acc = ""

      try {
        const res = await apiFetch(`${API}/system/v0/ta/chat/stream/${taskId}`, {
          signal: ctrl.signal,
          headers: { Accept: "text/event-stream" },
        })
        if (!res.ok || !res.body) {
          await recoverTask(taskId, ctrl.signal)
          return
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          // SSE frames are separated by a blank line
          let sep
          while ((sep = buf.indexOf("\n\n")) !== -1) {
            const frame = buf.slice(0, sep)
            buf = buf.slice(sep + 2)
            const line = frame.split("\n").find((l) => l.startsWith("data:"))
            if (!line) continue

            const evt = JSON.parse(line.slice(5).trim())
            if (evt.type === "step") {
              setThought({ thought: evt.label, agentName: "TA Agent" })
            } else if (evt.type === "token") {
              acc += evt.text
              setPartial(acc)
            } else if (evt.type === "done") {
              completeTask(taskId, evt.message ?? acc, evt.ui_action ?? null)
              return
            } else if (evt.type === "error") {
              failTask(evt.error ?? "TA workflow failed.")
              return
            }
          }
        }

        await recoverTask(taskId, ctrl.signal)
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return
        await recoverTask(taskId, ctrl.signal)
      }
    },
    [apiFetch, completeTask, failTask, recoverTask]
  )

  const submit = useCallback(
    async (userInput: string) => {
      if (!sessionId) return
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      setState("polling")
      setResult(null)
      setError(null)
      setThought(null)
      setPartial("")

      try {
        const res = await apiFetch(`${API}/system/v0/ta/chat`, {
          method: "POST",
          signal: ctrl.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            user_input: userInput,
            language,
          }),
        })

        if (!res.ok) {
          setState("fail")
          setError(`Could not reach TA (${res.status})`)
          return
        }

        const { task_id } = await res.json()
        if (ctrl.signal.aborted || !mountedRef.current) return
        await streamTask(task_id)
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return
        setState("fail")
        setError(err instanceof Error ? err.message : "Failed to send message")
      }
    },
    [apiFetch, sessionId, language, streamTask]
  )

  const resumeTask = useCallback(async (taskId: string) => {
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      const res = await apiFetch(`${API}/system/v0/ta/chat/status/${encodeURIComponent(taskId)}`, { signal: ctrl.signal })
      if (ctrl.signal.aborted || !mountedRef.current) return
      if (res.status === 404) {
        failTask("Lượt trả lời này đã gián đoạn. Bạn có thể gửi câu hỏi mới.")
        return
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const task = await res.json() as TaskStatusResponse
      if (task.status === "finished" && task.result) {
        completeTask(taskId, task.result.message, task.result.ui_action ?? null)
      } else if (task.status === "working") {
        setState("polling")
        await streamTask(taskId)
      } else {
        failTask(task.error ?? "Lượt trả lời đã thất bại.")
      }
    } catch (err) {
      if ((err as Error)?.name === "AbortError") return
      await recoverTask(taskId, ctrl.signal)
    }
  }, [apiFetch, completeTask, failTask, recoverTask, streamTask])

  return { state, result, error, thought, partial, submit, resumeTask, reset }
}
