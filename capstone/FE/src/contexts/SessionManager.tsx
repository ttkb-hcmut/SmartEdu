"use client"

import { useEffect, useRef } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { listSessions, newSession, resumeSession } from "@/lib/study"

export function SessionManager({ children }: { children: React.ReactNode }) {
  const { apiFetch, accessToken, setSessionId, setSessionReady } = useAuth()
  const inflightRef = useRef<Promise<string> | null>(null)

  useEffect(() => {
    if (!accessToken) return
    let cancelled = false

    async function restore(): Promise<string> {
      const fromUrl = window.location.pathname === "/chat"
        ? new URLSearchParams(window.location.search).get("session") : null
      const saved = window.localStorage.getItem("smartedu_session")
      async function firstAvailable(candidates: (string | null)[]) {
        for (const candidate of new Set(candidates.filter((value): value is string => !!value))) {
          try { return (await resumeSession(apiFetch, candidate)).session_id }
          catch { continue }
        }
        return null
      }
      let id = await firstAvailable([fromUrl, saved])
      if (!id) {
        const recent = await listSessions(apiFetch)
        id = await firstAvailable(recent.items.map((item) => item.id))
      }
      if (!id) id = (await newSession(apiFetch)).session_id
      return id
    }

    const pending = inflightRef.current ?? restore()
    inflightRef.current = pending
    pending.then((id) => {
      if (cancelled) return
      window.localStorage.setItem("smartedu_session", id)
      setSessionId(id)
    }).catch(() => {
      if (!cancelled) setSessionId(null)
    }).finally(() => {
      if (inflightRef.current === pending) inflightRef.current = null
      if (!cancelled) setSessionReady(true)
    })
    return () => { cancelled = true }
  }, [accessToken, apiFetch, setSessionId, setSessionReady])

  useEffect(() => {
    if (accessToken) return
    inflightRef.current = null
    setSessionReady(false)
  }, [accessToken, setSessionReady])

  return <>{children}</>
}
