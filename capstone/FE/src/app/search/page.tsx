"use client"

import { useEffect, useState, type FormEvent } from "react"
import Link from "next/link"
import { TopBar } from "@/components/layout/TopBar"
import { useAuth } from "@/contexts/AuthContext"
import { parseDocumentTarget } from "@/lib/normalise"
import { searchMaterials, type SearchHit } from "@/lib/study"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"

function Results({ title, hits }: { title: string; hits: SearchHit[] }) {
  return <section className="space-y-3">
    <h2 className="text-lg font-semibold">{title} <span className="text-sm font-normal text-[var(--ink-muted)]">({hits.length})</span></h2>
    {!hits.length && <p className="text-sm text-[var(--ink-muted)]">Không có kết quả.</p>}
    {hits.map((hit) => {
      const target = parseDocumentTarget(hit.document, hit.page)
      const href = target ? `/chat?document=${encodeURIComponent(hit.document!)}&page=${target.page}` : null
      return <article key={hit.id} className="rounded-lg border p-4" style={{ borderColor: "var(--se-border)" }}>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div><h3 className="font-medium">{hit.title}</h3><p className="text-xs text-[var(--ink-muted)]">{hit.course || "Chưa rõ môn"} · Điểm {Number(hit.score ?? 0).toFixed(2)}</p></div>
          {href ? <Link href={href} className="rounded bg-[var(--se-accent-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--se-accent)]">Mở PDF · trang {target!.page}</Link>
            : <span className="text-xs text-[var(--ink-muted)]">Chưa có PDF</span>}
        </div>
        <p className="mt-2 text-sm text-[var(--ink-muted)]">{hit.snippet}</p>
      </article>
    })}
  </section>
}

export default function SearchPage() {
  const { apiFetch } = useAuth()
  const [courses, setCourses] = useState<string[]>([])
  const [query, setQuery] = useState("")
  const [course, setCourse] = useState("")
  const [result, setResult] = useState<{ concepts: SearchHit[]; passages: SearchHit[] } | null>(null)
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle")

  useEffect(() => {
    apiFetch(`${API}/system/v0/knowledge/courses`).then((response) => response.json())
      .then((data) => setCourses(data.courses ?? [])).catch(() => {})
  }, [apiFetch])

  async function submit(event: FormEvent) {
    event.preventDefault()
    const trimmed = query.trim()
    if (trimmed.length < 2 || trimmed.length > 200) { setStatus("error"); return }
    setStatus("loading")
    try { setResult(await searchMaterials(apiFetch, trimmed, course)); setStatus("idle") }
    catch { setStatus("error") }
  }

  return <div className="min-h-dvh" style={{ background: "var(--bg)", color: "var(--ink)" }}>
    <TopBar title="Tìm kiếm tài liệu" backHref="/chat" />
    <main className="mx-auto max-w-4xl space-y-7 p-4 md:p-8">
      <div><h1 className="text-2xl font-semibold">Tìm trong kho học liệu</h1><p className="text-sm text-[var(--ink-muted)]">Khái niệm và đoạn sách liên quan đến câu hỏi của bạn.</p></div>
      <form onSubmit={submit} className="flex flex-col gap-3 sm:flex-row">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Nhập ít nhất 2 ký tự…" aria-label="Nội dung tìm kiếm"
          className="min-w-0 flex-1 rounded border bg-[var(--surface)] px-3 py-2" style={{ borderColor: "var(--se-border)" }} />
        <select value={course} onChange={(event) => setCourse(event.target.value)} aria-label="Lọc theo môn"
          className="rounded border bg-[var(--surface)] px-3 py-2" style={{ borderColor: "var(--se-border)" }}>
          <option value="">Tất cả môn</option>{courses.map((name) => <option key={name}>{name}</option>)}
        </select>
        <button type="submit" disabled={status === "loading"} className="rounded bg-[var(--se-primary)] px-5 py-2 font-medium text-white disabled:opacity-60">Tìm kiếm</button>
      </form>
      {status === "loading" && <p role="status">Đang tìm kiếm…</p>}
      {status === "error" && <p role="alert">Không thể tìm kiếm. Kiểm tra từ khóa hoặc thử lại sau.</p>}
      {result && status !== "loading" && <div className="space-y-8"><Results title="Khái niệm" hits={result.concepts} /><Results title="Đoạn sách" hits={result.passages} /></div>}
    </main>
  </div>
}
