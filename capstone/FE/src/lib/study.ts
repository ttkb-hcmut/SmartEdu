import type { ApiClient } from "@/lib/api"
import type { UiActionPayload } from "@/lib/normalise"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"
type ApiFetch = ApiClient["apiFetch"]

export interface SessionSummary {
  id: string
  name: string
  preview: string
  updated_at: string | null
}

export interface StudyMessage {
  id: string
  role: "user" | "ta"
  content: string
  timestamp: string | null
  ui_action: UiActionPayload | null
}

export interface SessionDetail {
  id: string
  messages: StudyMessage[]
  pending_chat_id: string | null
}

export interface RoadmapConcept {
  name: string
  type?: string
  description?: string
  requires: string[]
  mastery: number
  status: string
}

export interface RoadmapData {
  course: string
  topics: { name: string; concepts: RoadmapConcept[] }[]
  orphan_concepts: RoadmapConcept[]
  progress: { has_personal_path: boolean; mastered: number; in_progress: number; total: number }
}

export interface SearchHit {
  id: string
  course: string | null
  title: string
  snippet: string
  score: number
  document: string | null
  page: number | null
}

async function json<T>(apiFetch: ApiFetch, url: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(`${API}${url}`, init)
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json() as Promise<T>
}

const student = "/system/v0/student"

export function listSessions(apiFetch: ApiFetch, q = "", offset = 0) {
  const params = new URLSearchParams({ limit: "30", offset: String(offset), q })
  return json<{ items: SessionSummary[]; total: number }>(apiFetch, `${student}/sessions?${params}`)
}

export function readSession(apiFetch: ApiFetch, id: string) {
  return json<SessionDetail>(apiFetch, `${student}/sessions/${encodeURIComponent(id)}`)
}

export function resumeSession(apiFetch: ApiFetch, id: string) {
  return json<{ session_id: string }>(apiFetch, `${student}/sessions/${encodeURIComponent(id)}/resume`, { method: "POST" })
}

export function newSession(apiFetch: ApiFetch) {
  return json<{ session_id: string }>(apiFetch, `${student}/session/start`, { method: "POST" })
}

export function readRoadmap(apiFetch: ApiFetch, course: string) {
  return json<RoadmapData>(apiFetch, `/system/v0/ta/roadmap/${encodeURIComponent(course)}`)
}

export function searchMaterials(apiFetch: ApiFetch, q: string, course = "") {
  const params = new URLSearchParams({ q })
  if (course) params.set("course", course)
  return json<{ concepts: SearchHit[]; passages: SearchHit[] }>(apiFetch, `/system/v0/knowledge/search?${params}`)
}
