import { NextRequest, NextResponse } from "next/server"

const PROTECTED = ["/chat", "/settings", "/admin", "/search", "/roadmap"]

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl

  const isProtected = PROTECTED.some((p) => pathname.startsWith(p))
  if (!isProtected) return NextResponse.next()

  const hasRefreshCookie = request.cookies.has("refresh_token")
  if (!hasRefreshCookie) {
    const loginUrl = request.nextUrl.clone()
    loginUrl.pathname = "/login"
    loginUrl.search = ""
    return NextResponse.redirect(loginUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/chat/:path*", "/settings/:path*", "/admin/:path*", "/search/:path*", "/roadmap/:path*"],
}
