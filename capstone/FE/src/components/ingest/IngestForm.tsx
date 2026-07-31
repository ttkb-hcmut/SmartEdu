"use client"

import { useState } from "react"
import { toast } from "sonner"
import { FileDropzone } from "./FileDropzone"
import { UploadProgress, type FileProgress } from "./UploadProgress"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { useAuth } from "@/contexts/AuthContext"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"
const VIDEO_EXTS = [".mp4", ".mkv", ".webm", ".mp3", ".m4a", ".wav"]

type UploadTarget = {
  file_name: string
  url: string
}

function parseTargets(body: unknown, fileNames: Set<string>): UploadTarget[] {
  if (!body || typeof body !== "object" || !("targets" in body)) {
    throw new Error("Invalid upload URL response")
  }
  const targets = (body as { targets: unknown }).targets
  if (!Array.isArray(targets)) throw new Error("Invalid upload URL response")
  if (!targets.every((target): target is UploadTarget =>
    !!target && typeof target === "object" &&
    typeof (target as UploadTarget).file_name === "string" &&
    typeof (target as UploadTarget).url === "string"
  )) throw new Error("Invalid upload URL response")

  const names = new Set(targets.map((target) => target.file_name))
  if (names.size !== targets.length || names.size !== fileNames.size ||
      [...fileNames].some((name) => !names.has(name))) {
    throw new Error("Upload URL response does not match selected files")
  }
  return targets
}

export function IngestForm() {
  const { apiFetch } = useAuth()
  const [courseName, setCourseName] = useState("")
  const [slides, setSlides] = useState<File[]>([])
  const [textbooks, setTextbooks] = useState<File[]>([])
  const [videos, setVideos] = useState<File[]>([])
  const [progress, setProgress] = useState<FileProgress[]>([])
  const [submitting, setSubmitting] = useState(false)

  function updateProgress(name: string, patch: Partial<FileProgress>) {
    setProgress((prev) =>
      prev.map((f) => (f.name === name ? { ...f, ...patch } : f))
    )
  }

  async function uploadFile(file: File, url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open("PUT", url)
      xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream")

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100)
          updateProgress(file.name, { progress: pct })
        }
      })

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          updateProgress(file.name, { status: "done", progress: 100 })
          resolve()
        } else {
          updateProgress(file.name, { status: "error", error: `HTTP ${xhr.status}` })
          reject(new Error(`Upload failed: ${xhr.status}`))
        }
      })

      xhr.addEventListener("error", () => {
        updateProgress(file.name, { status: "error", error: "Network error" })
        reject(new Error("Network error"))
      })

      xhr.send(file)
    })
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!courseName.trim()) return
    const allFiles = [...slides, ...textbooks, ...videos]
    if (allFiles.length === 0) {
      toast.error("Vui lòng thêm ít nhất một file.")
      return
    }
    if (new Set(allFiles.map((file) => file.name)).size !== allFiles.length) {
      toast.error("Tên file phải duy nhất trong một lần nạp dữ liệu.")
      return
    }

    setSubmitting(true)
    setProgress(
      allFiles.map((f) => ({ name: f.name, status: "pending", progress: 0 }))
    )

    try {
      // Step 1: Get presigned upload URLs
      const urlRes = await apiFetch(`${API}/system/v0/knowledge/upload-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          course_name: courseName.trim(),
          file_names: allFiles.map((f) => f.name),
        }),
      })
      if (!urlRes.ok) throw new Error(`Failed to get upload URLs (${urlRes.status})`)
      const fileMap = new Map(allFiles.map((file) => [file.name, file]))
      const targets = parseTargets(await urlRes.json(), new Set(fileMap.keys()))

      // Step 2: Upload each file directly to MinIO
      setProgress((prev) =>
        prev.map((f) => ({ ...f, status: "uploading" as const }))
      )
      await Promise.all(
        targets.map(({ file_name, url }) => {
          const file = fileMap.get(file_name)
          if (!file) throw new Error(`Missing local file for ${file_name}`)
          return uploadFile(file, url)
        })
      )

      // Step 3: Trigger ingestion
      const ingestRes = await apiFetch(`${API}/system/v0/knowledge/ingest-course`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          course_name: courseName.trim(),
          slide_files: slides.map((f) => f.name),
          textbook_files: textbooks.map((f) => f.name),
          video_files: videos.map((f) => f.name),
          reset: true,
        }),
      })
      if (!ingestRes.ok) throw new Error(`Ingest failed (${ingestRes.status})`)

      toast.success("Đang xử lý tài liệu", {
        description: "Quá trình nạp dữ liệu đang chạy nền. Kiểm tra server log để theo dõi.",
        duration: 8000,
      })
      setCourseName("")
      setSlides([])
      setTextbooks([])
      setVideos([])
      setProgress([])
    } catch (err) {
      toast.error("Nạp dữ liệu thất bại", {
        description: err instanceof Error ? err.message : "Lỗi không xác định",
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-1.5">
        <label
          htmlFor="course-name"
          className="block text-sm font-medium"
          style={{ color: "var(--ink)" }}
        >
          Tên môn học
        </label>
        <Input
          id="course-name"
          placeholder="VD: MachineLearning"
          value={courseName}
          onChange={(e) => setCourseName(e.target.value)}
          required
          disabled={submitting}
        />
      </div>

      <FileDropzone
        label="Slides (PDF)"
        files={slides}
        onFilesChange={setSlides}
      />

      <FileDropzone
        label="Giáo trình (PDF)"
        files={textbooks}
        onFilesChange={setTextbooks}
      />

      <FileDropzone
        label="Video / audio"
        files={videos}
        onFilesChange={setVideos}
        extensions={VIDEO_EXTS}
      />

      {progress.length > 0 && (
        <UploadProgress files={progress} />
      )}

      <Button
        type="submit"
        className="w-full"
        disabled={submitting || !courseName.trim()}
      >
        {submitting && <Spinner size="sm" className="mr-1.5" />}
        {submitting ? "Đang tải lên…" : "Nạp tài liệu"}
      </Button>
    </form>
  )
}
