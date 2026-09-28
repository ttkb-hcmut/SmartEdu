"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react"
import { useRouter } from "next/navigation"
import { createApiClient } from "@/lib/api"

type Language = "vn" | "eng"

interface AuthState {
  accessToken: string | null
  isAdmin: boolean
  language: Language
  sessionId: string | null
  sessionReady: boolean
}

interface AuthContextValue extends AuthState {
  login: (token: string, isAdmin: boolean) => void
  logout: () => Promise<void>
  setLanguage: (lang: Language) => void
  setSessionId: (id: string | null) => void
  setSessionReady: (ready: boolean) => void
  apiFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [state, setState] = useState<AuthState>({
    accessToken: null,
    isAdmin: false,
    language: "vn",
    sessionId: null,
    sessionReady: false,
  })

  // tokenRef keeps apiFetch closures from going stale when accessToken changes
  const tokenRef = useRef<string | null>(null)

  const handleAuthFailure = useCallback(() => {
    setState({ accessToken: null, isAdmin: false, language: "vn", sessionId: null, sessionReady: false })
    tokenRef.current = null
    window.localStorage.removeItem("smartedu_session")
    router.push("/login")
  }, [router])

  const handleTokenRefresh = useCallback((newToken: string) => {
    tokenRef.current = newToken
    setState((s) => ({ ...s, accessToken: newToken }))
  }, [])

  const apiFetch = useCallback(
    (input: RequestInfo | URL, init?: RequestInit) =>
      createApiClient(() => tokenRef.current, handleTokenRefresh, handleAuthFailure).apiFetch(input, init),
    [handleTokenRefresh, handleAuthFailure]
  )

  const login = useCallback((token: string, isAdmin: boolean) => {
    tokenRef.current = token
    setState((s) => ({ ...s, accessToken: token, isAdmin }))
  }, [])

  const logout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" })
    } catch {
      // best-effort
    }
    setState({ accessToken: null, isAdmin: false, language: "vn", sessionId: null, sessionReady: false })
    tokenRef.current = null
    window.localStorage.removeItem("smartedu_session")
    router.push("/login")
  }, [router])

  const setLanguage = useCallback((lang: Language) => {
    setState((s) => ({ ...s, language: lang }))
  }, [])

  const setSessionId = useCallback((id: string | null) => {
    setState((s) => ({ ...s, sessionId: id }))
  }, [])

  const setSessionReady = useCallback((ready: boolean) => {
    setState((s) => ({ ...s, sessionReady: ready }))
  }, [])

  // On mount: try to restore session via the httpOnly refresh cookie
  useEffect(() => {
    async function restore() {
      try {
        const res = await fetch("/api/auth/refresh", { method: "POST" })
        if (!res.ok) return
        const data = await res.json()
        const token: string = data.access_token
        const isAdmin: boolean = data.is_admin ?? false
        setState((s) => ({ ...s, accessToken: token, isAdmin }))
        tokenRef.current = token

        if (window.location.pathname === "/login" || window.location.pathname === "/register") {
          router.push("/chat")
        }

        // Load language preference
        const profile = await fetch("/api/profile", {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (profile.ok) {
          const p = await profile.json()
          if (p.language) setState((s) => ({ ...s, language: p.language }))
        }
      } catch {
        // No valid session — stay logged out
      }
    }
    restore()
  }, [router])

  return (
    <AuthContext.Provider
      value={{ ...state, login, logout, setLanguage, setSessionId, setSessionReady, apiFetch }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>")
  return ctx
}
