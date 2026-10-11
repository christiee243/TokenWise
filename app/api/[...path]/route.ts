/**
 * Next.js API Route Proxy – forwards all /api/* requests to the
 * FastAPI backend running on http://127.0.0.1:8000.
 *
 * This is needed because client-side fetch() to an absolute
 * http://127.0.0.1:8000 URL fails when the page is rendered via the
 * Cloudflare Worker sandbox (vinext), and relative URLs are intercepted
 * by the worker before reaching the Vite proxy.
 *
 * Route: /api/[...path]  →  http://127.0.0.1:8000/api/[...path]
 */

import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE = "http://127.0.0.1:8000";

// Allow large PDF uploads (up to 30 MB) and use Node.js runtime for binary body handling
export const runtime = "nodejs";
export const maxDuration = 60;


async function proxyRequest(req: NextRequest, path: string[]): Promise<NextResponse> {
  const targetPath = "/api/" + path.join("/");
  const targetUrl = FASTAPI_BASE + targetPath + (req.nextUrl.search || "");

  try {
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: (() => {
        const h = new Headers(req.headers);
        h.delete("host");
        return h;
      })(),
      body: req.method !== "GET" && req.method !== "HEAD" ? await req.arrayBuffer() : undefined,
      // @ts-expect-error – Node.js fetch needs this to follow redirects
      redirect: "follow",
    });

    const responseHeaders = new Headers(response.headers);
    // Remove encoding headers that can cause issues
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("transfer-encoding");

    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error: any) {
    console.error("[API Proxy] Error forwarding to FastAPI:", error?.message);
    return NextResponse.json(
      { detail: "FastAPI backend is unreachable. Ensure the backend server is running on port 8000." },
      { status: 503 }
    );
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyRequest(req, path);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyRequest(req, path);
}

export async function PUT(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyRequest(req, path);
}

export async function DELETE(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyRequest(req, path);
}

export async function PATCH(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  return proxyRequest(req, path);
}
