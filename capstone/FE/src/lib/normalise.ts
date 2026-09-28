/**
 * Normalises the two ui_action shapes the backend can emit into a single
 * canonical UiAction. All PDF navigation in the UI flows through this.
 *
 * Shape A (teach.py / smart_edu.py):
 *   { navigate_page: number, document: "minio://bucket/courses/{course}/{topic}/..." }
 *
 * Shape B (FEToPage tool):
 *   { action: "NAVIGATE_PDF", destination: "minio://...", page: number }
 */

export interface UiAction {
  action: "NAVIGATE_PDF"
  course: string
  topic: string
  destination: string
  page: number
}

export interface Citation {
  uri: string
  document: string | null
  page: number | null
}

export interface CitationAction {
  citations: Citation[]
}

export function parseDocumentTarget(document: string | null, page: number | null) {
  if (!document || !Number.isInteger(page) || !page || page < 1) return null
  const parts = document.split("/")
  if (parts.length !== 3 || parts.some((part) => !part || part === "." || part === "..")) return null
  if (parts[1] === "_raw" && parts[2].toLowerCase().endsWith(".pdf")) {
    return { course: parts[0], topic: null, rawFile: parts[2], page }
  }
  if (parts[2] === "page.pdf") {
    return { course: parts[0], topic: parts[1], rawFile: null, page }
  }
  return null
}

export type UiActionPayload = UiAction | CitationAction

interface RawShapeA {
  navigate_page: number
  document: string
}

interface RawShapeB {
  action: "NAVIGATE_PDF"
  destination: string
  page: number
}

function isShapeA(raw: unknown): raw is RawShapeA {
  return (
    typeof raw === "object" &&
    raw !== null &&
    "navigate_page" in raw &&
    "document" in raw
  )
}

function isShapeB(raw: unknown): raw is RawShapeB {
  return (
    typeof raw === "object" &&
    raw !== null &&
    (raw as RawShapeB).action === "NAVIGATE_PDF" &&
    "destination" in raw &&
    "page" in raw
  )
}

/**
 * Extracts {course, topic} from a minio URI.
 * Expected path after stripping the "minio://bucket/" prefix:
 *   courses/{course}/{topic}/...
 */
function parseMinioDest(uri: string): { course: string; topic: string } {
  // Strip minio://bucket/ prefix
  const withoutScheme = uri.replace(/^minio:\/\/[^/]+\//, "")
  const parts = withoutScheme.split("/")
  // parts[0] = "courses", parts[1] = course, parts[2] = topic
  const course = parts[1] ?? ""
  const topic = parts[2] ?? ""
  return { course, topic }
}

/**
 * Normalises any raw ui_action value from the backend into a UiAction,
 * or returns null if the value is absent or unrecognised.
 */
export function normaliseUiAction(raw: unknown): UiActionPayload | null {
  if (raw == null) return null

  if (typeof raw === "object" && "citations" in raw && Array.isArray(raw.citations)) {
    return {
      citations: raw.citations.filter((item: unknown): item is Citation =>
        typeof item === "object" && item !== null &&
        typeof (item as Citation).uri === "string" &&
        ((item as Citation).document === null || typeof (item as Citation).document === "string") &&
        ((item as Citation).page === null || Number.isInteger((item as Citation).page))
      ),
    }
  }

  if (isShapeA(raw)) {
    const { course, topic } = parseMinioDest(raw.document)
    return {
      action: "NAVIGATE_PDF",
      course,
      topic,
      destination: raw.document,
      page: raw.navigate_page,
    }
  }

  if (isShapeB(raw)) {
    const { course, topic } = parseMinioDest(raw.destination)
    return {
      action: "NAVIGATE_PDF",
      course,
      topic,
      destination: raw.destination,
      page: raw.page,
    }
  }

  return null
}
