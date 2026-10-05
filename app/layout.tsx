import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tokenwise — PDF to optimized Markdown",
  description: "Convert dense PDFs into clean, structured, token-efficient Markdown.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
