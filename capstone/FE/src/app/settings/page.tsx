"use client"

import { useSyncExternalStore } from "react"
import { useTheme } from "next-themes"
import { toast } from "sonner"
import { useAuth } from "@/contexts/AuthContext"
import { Button } from "@/components/ui/button"
import { TopBar } from "@/components/layout/TopBar"
import { cn } from "@/lib/utils"

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-3">
      <h2
        className="text-sm font-semibold"
        style={{ color: "var(--ink)" }}
      >
        {title}
      </h2>
      {children}
      <hr style={{ borderColor: "var(--se-border)" }} />
    </section>
  )
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { language, setLanguage, apiFetch } = useAuth()
  const mounted = useSyncExternalStore(() => () => {}, () => true, () => false)

  async function updateLanguage(lang: "vn" | "eng") {
    setLanguage(lang)
    try {
      await apiFetch("/api/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: lang }),
      })
    } catch {
      toast.error("Không thể lưu cài đặt ngôn ngữ.")
    }
  }

  const langOptions: { value: "vn" | "eng"; label: string }[] = [
    { value: "vn", label: "Tiếng Việt" },
    { value: "eng", label: "English" },
  ]

  const themeOptions: { value: string; label: string }[] = [
    { value: "light", label: "Sáng" },
    { value: "dark", label: "Tối" },
    { value: "system", label: "Hệ thống" },
  ]

  return (
    <div className="flex min-h-dvh flex-col" style={{ backgroundColor: "var(--bg)" }}>
      <TopBar title="Cài đặt" backHref="/chat" />

      <main className="mx-auto w-full max-w-[560px] space-y-6 px-6 py-12">
        <h1
          className="text-2xl font-semibold"
          style={{ color: "var(--ink)" }}
        >
          Cài đặt
        </h1>

        <Section title="Ngôn ngữ phản hồi">
          <div className="flex gap-2">
            {langOptions.map((opt) => (
              <Button
                key={opt.value}
                variant="outline"
                size="sm"
                onClick={() => updateLanguage(opt.value)}
                className={cn(mounted && language === opt.value && "ring-2 ring-[--ring]")}
                style={
                  mounted && language === opt.value
                    ? { borderColor: "var(--se-accent)", color: "var(--se-accent)" }
                    : undefined
                }
              >
                {opt.label}
              </Button>
            ))}
          </div>
          <p className="text-xs" style={{ color: "var(--ink-muted)" }}>
            Ngôn ngữ trợ lý sẽ dùng khi trả lời câu hỏi của bạn.
          </p>
        </Section>

        <Section title="Giao diện">
          <div className="flex gap-2">
            {themeOptions.map((opt) => (
              <Button
                key={opt.value}
                variant="outline"
                size="sm"
                onClick={() => setTheme(opt.value)}
                className={cn(mounted && theme === opt.value && "ring-2 ring-[--ring]")}
                style={
                  mounted && theme === opt.value
                    ? { borderColor: "var(--se-accent)", color: "var(--se-accent)" }
                    : undefined
                }
              >
                {opt.label}
              </Button>
            ))}
          </div>
        </Section>
      </main>
    </div>
  )
}
