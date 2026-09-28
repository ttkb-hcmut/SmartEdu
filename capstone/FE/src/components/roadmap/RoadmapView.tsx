"use client"

import { useState } from "react"
import type { RoadmapConcept, RoadmapData } from "@/lib/study"

const stateColor: Record<string, string> = {
  mastered: "var(--success)", in_progress: "var(--se-accent)",
}

export function RoadmapView({ data }: { data: RoadmapData }) {
  const groups = [...data.topics, ...(data.orphan_concepts.length ? [{ name: "Khái niệm khác", concepts: data.orphan_concepts }] : [])]
  const [selected, setSelected] = useState<RoadmapConcept | null>(null)
  const layout = (() => {
    const points = new Map<string, { x: number; y: number }>()
    groups.forEach((group, column) => group.concepts.forEach((node, row) => {
      points.set(node.name.toLocaleLowerCase(), { x: 145 + column * 260, y: 88 + row * 92 })
    }))
    return points
  })()
  const width = Math.max(640, groups.length * 260 + 40)
  const height = Math.max(230, Math.max(...groups.map((g) => g.concepts.length), 0) * 92 + 70)

  return <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_280px]">
    <div className="overflow-x-auto rounded-lg border bg-[var(--surface)]" style={{ borderColor: "var(--se-border)" }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Sơ đồ khái niệm và kiến thức tiên quyết">
        <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="var(--ink-muted)" /></marker></defs>
        {groups.map((group, column) => <text key={group.name} x={55 + column * 260} y={34} fill="var(--ink)" fontSize="13" fontWeight="600">{group.name.slice(0, 30)}</text>)}
        {groups.flatMap((group) => group.concepts.flatMap((node) => node.requires.map((requirement) => {
          const from = layout.get(requirement.toLocaleLowerCase())
          const to = layout.get(node.name.toLocaleLowerCase())
          return from && to ? <line key={`${requirement}-${node.name}`} x1={from.x + 85} y1={from.y + 20}
            x2={to.x - 85} y2={to.y + 20} stroke="var(--ink-muted)" strokeWidth="1.5" markerEnd="url(#arrow)" /> : null
        })))}
        {groups.flatMap((group) => group.concepts.map((node) => {
          const point = layout.get(node.name.toLocaleLowerCase())!
          return <g key={node.name} tabIndex={0} role="button" aria-label={`Xem ${node.name}`}
            onClick={() => setSelected(node)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelected(node) } }}
            className="cursor-pointer outline-none focus-visible:opacity-70">
            <rect x={point.x - 85} y={point.y} width="170" height="48" rx="8" fill="var(--bg)"
              stroke={stateColor[node.status] ?? "var(--se-border)"} strokeWidth={selected?.name === node.name ? 3 : 2} />
            <title>{node.name}</title>
            <text x={point.x} y={point.y + 29} textAnchor="middle" fill="var(--ink)" fontSize="12" textLength={node.name.length > 20 ? 150 : undefined} lengthAdjust="spacingAndGlyphs">{node.name.slice(0, 24)}</text>
          </g>
        }))}
      </svg>
    </div>
    <aside className="rounded-lg border p-5" style={{ borderColor: "var(--se-border)" }}>
      {selected ? <div className="space-y-3">
        <h2 className="text-lg font-semibold">{selected.name}</h2>
        <p className="text-sm text-[var(--ink-muted)]">{selected.description || "Chưa có mô tả."}</p>
        <p className="text-sm">Trạng thái: {selected.status === "mastered" ? "Đã nắm vững" : selected.status === "in_progress" ? "Đang học" : "Chưa bắt đầu"}</p>
        {data.progress.has_personal_path && <p className="text-sm">Mastery: {selected.mastery}</p>}
        <div className="text-sm">Cần biết trước: {selected.requires.length ? selected.requires.join(", ") : "Không có"}</div>
      </div> : <p className="text-sm text-[var(--ink-muted)]">Chọn một khái niệm để xem chi tiết.</p>}
    </aside>
  </div>
}
