import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/contexts/AuthContext";
import { SessionManager } from "@/contexts/SessionManager";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "SmartEdu",
  description: "AI Teaching Assistant for Vietnamese university students",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body className="min-h-dvh antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <AuthProvider>
            <SessionManager>
              {children}
              <Toaster richColors closeButton />
            </SessionManager>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
