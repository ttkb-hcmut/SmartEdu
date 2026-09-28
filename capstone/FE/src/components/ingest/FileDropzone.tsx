"use client"

import { useRef, useState, type DragEvent } from "react"
import { Upload, X, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface FileDropzoneProps {
  label: string
  files: File[]
  onFilesChange: (files: File[]) => void
  extensions?: string[]
}

export function FileDropzone({ label, files, onFilesChange, extensions = [".pdf"] }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function addFiles(incoming: FileList | null) {
    if (!incoming) return
    const valid = Array.from(incoming).filter((f) =>
      extensions.some((ext) => f.name.toLowerCase().endsWith(ext))
    )
    onFilesChange([...files, ...valid])
  }

  function removeFile(index: number) {
    onFilesChange(files.filter((_, i) => i !== index))
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium" style={{ color: "var(--ink)" }}>
        {label}
      </p>

      {/* Drop target */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-sm border-2 border-dashed px-4 py-8 text-center transition-colors",
          dragging
            ? "border-[var(--se-accent)] bg-[var(--se-accent-subtle)]"
            : "border-[var(--se-border)] hover:border-[var(--se-accent)] hover:bg-[var(--se-accent-subtle)]"
        )}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        aria-label={`${label} — kéo thả hoặc nhấn để chọn`}
      >
        <Upload className="size-5" style={{ color: "var(--ink-muted)" }} />
        <p className="text-sm" style={{ color: "var(--ink-muted)" }}>
          Kéo thả {extensions.join(", ")} vào đây, hoặc{" "}
          <span style={{ color: "var(--se-accent)" }}>nhấn để chọn</span>
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={extensions.join(",")}
        multiple
        className="sr-only"
        onChange={(e) => addFiles(e.target.files)}
      />

      {/* File list */}
      {files.length > 0 && (
        <ul className="space-y-1">
          {files.map((file, i) => (
            <li
              key={`${file.name}-${i}`}
              className="flex items-center gap-2 rounded-xs px-2 py-1.5 text-sm"
              style={{ backgroundColor: "var(--surface)", color: "var(--ink)" }}
            >
              <FileText className="size-3.5 shrink-0" style={{ color: "var(--ink-muted)" }} />
              <span className="flex-1 truncate">{file.name}</span>
              <span className="font-mono text-xs" style={{ color: "var(--ink-muted)" }}>
                {(file.size / 1024).toFixed(0)} KB
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={(e) => { e.stopPropagation(); removeFile(i) }}
                aria-label={`Xóa ${file.name}`}
              >
                <X className="size-3" />
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
