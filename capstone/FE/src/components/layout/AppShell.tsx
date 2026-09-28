"use client"

import { useEffect } from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"

interface AppShellProps {
  sidebar?: React.ReactNode
  panel?: React.ReactNode
  children: React.ReactNode
  className?: string
  sidebarOpen?: boolean
  onCloseSidebar?: () => void
  mobileTab?: "chat" | "pdf"
}

export function AppShell({ sidebar, panel, children, className, sidebarOpen = false, onCloseSidebar, mobileTab = "chat" }: AppShellProps) {
  useEffect(() => {
    if (!sidebarOpen) return
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") onCloseSidebar?.() }
    window.addEventListener("keydown", close)
    return () => window.removeEventListener("keydown", close)
  }, [sidebarOpen, onCloseSidebar])

  return (
    <div className={cn("flex h-screen overflow-hidden", className)}>
      {sidebar && (
        <aside
          className="hidden w-60 shrink-0 flex-col border-r lg:flex"
          style={{
            backgroundColor: "var(--surface-2)",
            borderColor: "var(--se-border)",
          }}
        >
          {sidebar}
        </aside>
      )}

      {sidebar && sidebarOpen && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Điều hướng">
          <button type="button" aria-label="Đóng menu" onClick={onCloseSidebar} className="absolute inset-0 bg-black/40" />
          <aside className="relative flex h-full w-72 max-w-[85vw] flex-col border-r" style={{ backgroundColor: "var(--surface-2)", borderColor: "var(--se-border)" }}>
            <button type="button" aria-label="Đóng menu" onClick={onCloseSidebar} className="absolute right-2 top-3 z-10 rounded p-1"><X className="size-5" /></button>
            {sidebar}
          </aside>
        </div>
      )}

      <main className={cn("min-w-0 flex-1 flex-col overflow-hidden", mobileTab === "chat" ? "flex" : "hidden lg:flex")} style={{ backgroundColor: "var(--bg)" }}>
        {children}
      </main>

      {panel && (
        <div
          className={cn("min-w-0 flex-1 flex-col border-l lg:w-[420px] lg:flex-none", mobileTab === "pdf" ? "flex" : "hidden lg:flex")}
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--se-border)",
          }}
        >
          {panel}
        </div>
      )}
    </div>
  )
}
