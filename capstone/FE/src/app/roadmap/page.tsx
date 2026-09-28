"use client"

import { useEffect, useState } from "react"
import { TopBar } from "@/components/layout/TopBar"
import { RoadmapView } from "@/components/roadmap/RoadmapView"
import { useAuth } from "@/contexts/AuthContext"
import { readRoadmap, type RoadmapData } from "@/lib/study"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"

export default function RoadmapPage() {
  const { apiFetch } = useAuth()
  const [courses, setCourses] = useState<string[]>([])
  const [course, setCourse] = useState("")
  const [data, setData] = useState<RoadmapData | null>(null)
  const [state, setState] = useState<"loading" | "ready" | "error">("loading")

  useEffect(() => {
    apiFetch(`${API}/system/v0/knowledge/courses`).then((response) => {
      if (!response.ok) throw new Error()
      return response.json()
    }).then((result) => { setCourses(result.courses ?? []); setState("ready") }).catch(() => setState("error"))
  }, [apiFetch])
  useEffect(() => {
    if (!course) return
    let cancelled = false
    readRoadmap(apiFetch, course).then((result) => {
      if (!cancelled) { setData(result); setState("ready") }
    }).catch(() => { if (!cancelled) setState("error") })
    return () => { cancelled = true }
  }, [apiFetch, course])

  const progress = data?.progress
  return <div className="min-h-dvh" style={{ background: "var(--bg)", color: "var(--ink)" }}>
    <TopBar title="Roadmap & tiến độ" backHref="/chat" />
    <main className="mx-auto max-w-7xl space-y-6 p-4 md:p-8">
      <div><h1 className="text-2xl font-semibold">Roadmap học tập</h1><p className="text-sm text-[var(--ink-muted)]">Chọn môn để xem khái niệm và kiến thức tiên quyết.</p></div>
      <label className="block text-sm">Môn học
        <select value={course} onChange={(event) => { setCourse(event.target.value); setData(null); setState(event.target.value ? "loading" : "ready") }} className="mt-1 block w-full max-w-sm rounded border bg-[var(--surface)] p-2" style={{ borderColor: "var(--se-border)" }}>
          <option value="">Chọn môn học</option>{courses.map((name) => <option key={name}>{name}</option>)}
        </select>
      </label>
      {state === "loading" && <p role="status">Đang tải roadmap…</p>}
      {state === "error" && <p role="alert">Không thể tải roadmap. Hãy thử lại sau.</p>}
      {data && state === "ready" && <>
        <section className="rounded-lg border p-4" style={{ borderColor: "var(--se-border)" }}>
          {progress?.has_personal_path ? <p>Đã nắm vững {progress.mastered}/{progress.total} khái niệm · Đang học {progress.in_progress}</p>
            : <p>Chưa có lộ trình cá nhân. Đây là sơ đồ môn học; tiến độ sẽ hiện khi bạn bắt đầu học.</p>}
        </section>
        {data.topics.length || data.orphan_concepts.length ? <RoadmapView data={data} /> : <p>Chưa có khái niệm cho môn này.</p>}
      </>}
    </main>
  </div>
}
