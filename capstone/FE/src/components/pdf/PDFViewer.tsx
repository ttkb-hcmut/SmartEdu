"use client"

import { useEffect, useRef, useState } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import "react-pdf/dist/Page/AnnotationLayer.css"
import "react-pdf/dist/Page/TextLayer.css"
import { useAuth } from "@/contexts/AuthContext"
import { PDFSkeleton } from "./PDFSkeleton"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

// Initialise pdfjs worker — must run client-side only (this file is ssr:false)
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString()

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"

interface PDFViewerProps {
  course: string
  topic: string
  rawFile?: string
  page: number
  onPageChange: (page: number) => void
  className?: string
}

export function PDFViewer({ course, topic, rawFile, page, onPageChange, className }: PDFViewerProps) {
  const { apiFetch } = useAuth()
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [numPages, setNumPages] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [viewerWidth, setViewerWidth] = useState(392)
  const viewportRef = useRef<HTMLDivElement>(null)
  const prevBlobRef = useRef<string | null>(null)

  useEffect(() => {
    const element = viewportRef.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setViewerWidth(entry.contentRect.width))
    observer.observe(element)
    return () => observer.disconnect()
  }, [loading])

  useEffect(() => {
    if (!course || (!topic && !rawFile)) return
    let cancelled = false

    const path = rawFile
      ? `raw/${encodeURIComponent(course)}/${encodeURIComponent(rawFile)}`
      : `${encodeURIComponent(course)}/${encodeURIComponent(topic)}`
    apiFetch(`${API}/system/v0/knowledge/pdf/${path}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.blob()
      })
      .then((blob) => {
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        setBlobUrl(url)
        setLoading(false)
      })
      .catch((err) => {
        if (cancelled) return
        setFetchError(err.message ?? "Could not load PDF")
        setLoading(false)
      })

    return () => {
      cancelled = true
      // Revoke the previous blob URL to avoid memory leaks
      if (prevBlobRef.current) {
        URL.revokeObjectURL(prevBlobRef.current)
        prevBlobRef.current = null
      }
    }
  }, [course, topic, rawFile, apiFetch])

  // Track current blobUrl for cleanup
  useEffect(() => {
    if (blobUrl) prevBlobRef.current = blobUrl
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl)
    }
  }, [blobUrl])

  if (loading) return <PDFSkeleton className={className} />

  if (fetchError) {
    return (
      <div
        className={cn("flex flex-col items-center justify-center gap-2 p-8 text-sm", className)}
        style={{ color: "var(--ink-muted)" }}
      >
        <p>Không thể tải trang PDF.</p>
        <p className="font-mono text-xs" style={{ color: "var(--error)" }}>
          {fetchError}
        </p>
      </div>
    )
  }

  return (
    <div className={cn("flex h-full flex-col overflow-hidden", className)}>
      {/* PDF document */}
      <div ref={viewportRef} className="flex-1 overflow-y-auto px-4 py-4">
        <Document
          file={blobUrl}
          onLoadSuccess={({ numPages }) => {
            setNumPages(numPages)
            if (page > numPages) onPageChange(numPages)
          }}
          loading={<PDFSkeleton />}
          error={
            <p className="text-sm" style={{ color: "var(--error)" }}>
              Lỗi khi render PDF.
            </p>
          }
        >
          <Page
            pageNumber={page}
            width={Math.max(100, viewerWidth - 32)}
            renderTextLayer
            renderAnnotationLayer
          />
        </Document>
      </div>

      {/* Page navigation */}
      {numPages > 1 && (
        <div
          className="flex shrink-0 items-center justify-between border-t px-4 py-2"
          style={{ borderColor: "var(--se-border)", backgroundColor: "var(--surface)" }}
        >
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onPageChange(Math.max(1, page - 1))}
            disabled={page <= 1}
            aria-label="Trang trước"
          >
            <ChevronLeft className="size-4" />
          </Button>

          <span className="font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
            {page} / {numPages}
          </span>

          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onPageChange(Math.min(numPages, page + 1))}
            disabled={page >= numPages}
            aria-label="Trang sau"
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  )
}
