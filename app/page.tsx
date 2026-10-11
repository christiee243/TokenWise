"use client";

import { useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  ArrowUp,
  Check,
  Clipboard,
  Download,
  FileText,
  Key,
  Loader2,
  LogOut,
  RotateCcw,
  Search,
  Shield,
  Sparkles,
  UploadCloud,
  User,
  WandSparkles,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type FileDetails = {
  rawFile: File;
  name: string;
  size: string;
  pages: number;
};

type AnalysisResult = {
  markdown: string;
  original_tokens: number;
  optimized_tokens: number;
  tokens_saved: number;
  reduction_percent: number;
  page_count: number;
  section_count: number;
  table_count: number;
  chunk_count: number;
  is_scanned?: boolean;
  chunks?: Array<{
    chunk_id: number;
    text: string;
    section: string;
    tokens: number;
  }>;
};

type PromptResult = {
  optimized: string;
  original_tokens: number;
  optimized_tokens: number;
  tokens_saved: number;
  reduction_percent: number;
  changes: string[];
  method?: "tiktoken" | "approx";
  semantic_warning?: boolean;
};

type RetrievedChunk = {
  text: string;
  relevance_score: number;
  metadata: {
    chunk_id: number;
    section: string;
    tokens: number;
  };
};

type QueryResult = {
  query_analysis: {
    intent: string;
    complexity: string;
    keywords: string[];
  };
  direct_answer?: string;
  retrieved_chunks: RetrievedChunk[];
  num_chunks_retrieved: number;
  suggested_top_k: number;
};

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<FileDetails | null>(null);
  const [status, setStatus] = useState<"idle" | "working" | "done" | "error">(
    "idle",
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  // Account Menu & Modal state
  const [activeDialog, setActiveDialog] = useState<
    "profile" | "api_keys" | "security" | "how" | null
  >(null);
  const [userProfile, setUserProfile] = useState({
    name: "Demo User",
    email: "demo@tokenwise.local",
    role: "Demo account",
    organization: "Local demo workspace",
  });

  // Real data state
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null,
  );
  const [prompt, setPrompt] = useState("");
  const [promptResult, setPromptResult] = useState<PromptResult | null>(null);
  const [promptHistory, setPromptHistory] = useState<
    Array<{ id: string; original: string; result: PromptResult }>
  >([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [promptLoading, setPromptLoading] = useState(false);
  const [activeTool, setActiveTool] = useState<"pdf" | "prompt" | "qa">("pdf");

  // M2 RAG state
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [indexStatus, setIndexStatus] = useState<
    "idle" | "indexing" | "indexed" | "error"
  >("idle");
  const [queryText, setQueryText] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryHistory, setQueryHistory] = useState<
    Array<{ query: string; result: QueryResult }>
  >([]);

  const resetDocumentState = () => {
    setDocumentId(null);
    setIndexStatus("idle");
    setQueryText("");
    setQueryResult(null);
    setQueryHistory([]);
    setQueryError(null);
  };

  const chooseFile = (selected?: File) => {
    if (!selected) return;
    const maxSizeBytes = 30 * 1024 * 1024;
    const isPdf =
      selected.type === "application/pdf" ||
      (!selected.type && selected.name.toLowerCase().endsWith(".pdf"));
    if (!isPdf) {
      toast.error("Choose a PDF file to continue.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    if (selected.size > maxSizeBytes) {
      toast.error("PDF files must be 30 MB or smaller.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    resetDocumentState();
    if (inputRef.current) inputRef.current.value = "";
    setFile({
      rawFile: selected,
      name: selected.name,
      size: `${(selected.size / 1024 / 1024).toFixed(1)} MB`,
      pages: 1, // Will update accurately after backend analysis
    });
    setStatus("idle");
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  const analyze = async () => {
    if (!file || !file.rawFile) return;
    setStatus("working");
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file.rawFile);
    formData.append("compression", "Balanced");

    try {
      const response = await fetch(`${API_BASE}/api/analyze-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = (await response.json().catch(() => ({}))) as {
          detail?: string;
        };
        throw new Error(
          errorData.detail || `Server responded with ${response.status}`,
        );
      }

      const data: AnalysisResult = await response.json();
      setAnalysisResult(data);
      // Update page count from backend
      setFile((prev) => (prev ? { ...prev, pages: data.page_count } : null));
      setStatus("done");

      // Auto-index chunks for RAG (M2)
      if (data.chunks && data.chunks.length > 0) {
        autoIndexChunks(data.chunks, file!.name);
      }
    } catch (err: unknown) {
      console.error("PDF Analysis failed:", err);
      setErrorMessage(
        (err instanceof Error ? err.message : null) ||
          "Failed to connect to backend server. Make sure the FastAPI service is running.",
      );
      setStatus("error");
    }
  };

  // ── M2: Auto-index chunks after PDF analysis ──────────────────────────
  const autoIndexChunks = async (
    chunks: NonNullable<AnalysisResult["chunks"]>,
    filename: string,
  ) => {
    const baseId = filename
      .replace(/\.pdf$/i, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 40);
    const uniqueId =
      globalThis.crypto?.randomUUID?.() ??
      `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const docId = `${baseId || "document"}-${uniqueId}`;
    setIndexStatus("indexing");
    setDocumentId(null);
    setQueryResult(null);
    setQueryHistory([]);
    setQueryError(null);

    try {
      const response = await fetch(`${API_BASE}/api/index-document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: docId, chunks }),
      });

      if (!response.ok) {
        throw new Error("Failed to index document");
      }

      const data = (await response.json()) as { document_id: string };
      setDocumentId(data.document_id);
      setIndexStatus("indexed");
    } catch (err) {
      console.error("Indexing failed:", err);
      setIndexStatus("error");
      setQueryError("Could not index this PDF. Try analyzing it again.");
      toast.error("Could not index this PDF. Try analyzing it again.");
    }
  };

  // ── M2: Query the indexed document ────────────────────────────────────
  const handleQuery = async () => {
    if (!queryText.trim() || !documentId || queryLoading) return;
    setQueryLoading(true);
    setQueryError(null);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: documentId,
          query: queryText.trim(),
          top_k: null,
        }),
      });

      if (!response.ok) {
        throw new Error("Query failed");
      }

      const data: QueryResult = await response.json();
      setQueryResult(data);
      setQueryHistory((prev) => [
        ...prev,
        { query: queryText.trim(), result: data },
      ]);
      setQueryText("");
    } catch (err) {
      console.error("Query failed:", err);
      const message =
        err instanceof Error ? err.message : "Could not query this document.";
      setQueryError(message);
      toast.error(message);
    } finally {
      setQueryLoading(false);
    }
  };

  const copyMarkdown = async () => {
    const textToCopy = analysisResult?.markdown || "";
    if (!textToCopy) return;
    await navigator.clipboard?.writeText(textToCopy);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const exportMarkdown = () => {
    const text = analysisResult?.markdown;
    if (!text) return;
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = file?.name?.replace(/\.pdf$/i, "") || "document";
    link.href = url;
    link.download = `${baseName}-optimized.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleOptimizePrompt = async (overridePrompt?: unknown) => {
    const rawText = (
      typeof overridePrompt === "string" ? overridePrompt : prompt
    ).trim();
    if (!rawText || promptLoading) return;
    setPromptLoading(true);

    const messageId =
      Date.now().toString() + Math.random().toString(36).slice(2, 6);

    try {
      const response = await fetch(`${API_BASE}/api/optimize-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: rawText }),
      });

      if (!response.ok) {
        throw new Error("Failed to optimize prompt");
      }

      const data: PromptResult = await response.json();
      setPrompt("");
      setPromptResult(data);
      setPromptHistory((prev) => [
        ...prev,
        { id: messageId, original: rawText, result: data },
      ]);
    } catch (err: unknown) {
      console.error("Prompt optimization failed:", err);
      toast.error(
        err instanceof Error
          ? err.message
          : "Prompt optimization failed. Your prompt is still available to retry.",
      );
    } finally {
      setPromptLoading(false);
    }
  };

  const copyOptimizedPrompt = async (id?: string, text?: string) => {
    const textToCopy = text || promptResult?.optimized;
    if (!textToCopy) return;
    await navigator.clipboard?.writeText(textToCopy);
    setCopiedId(id || "latest");
    window.setTimeout(() => {
      setCopiedId(null);
    }, 1600);
  };

  return (
    <main className="h-dvh overflow-hidden bg-[#f4f5f0] text-[#101827]">
      <header className="border-b border-[#101827]/10 bg-[#f4f5f0]/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-5 sm:px-8">
          <a
            href="#"
            className="flex items-center gap-3"
            aria-label="Tokenwise home"
          >
            <span className="grid size-9 place-items-center rounded-[10px] bg-[#101827] text-[#d8ff4f]">
              <span className="text-lg font-black tracking-[-0.16em]">T/</span>
            </span>
            <span className="text-[19px] font-bold tracking-[-0.04em]">
              Tokenwise
            </span>
          </a>
          <nav
            className="hidden items-center gap-8 text-sm font-medium text-[#566071] md:flex"
            aria-label="Main navigation"
          >
            <button
              type="button"
              className="text-[#101827]"
              onClick={() => setActiveTool("pdf")}
            >
              Workspace
            </button>
            <button
              type="button"
              className="transition-colors hover:text-[#101827]"
              onClick={() => setActiveDialog("how")}
            >
              How it works
            </button>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#101827]/5 px-2.5 py-1 text-xs font-semibold text-[#6a7516]">
              <span className="size-2 rounded-full bg-[#6a7516] animate-pulse" />{" "}
              Backend Active
            </span>
          </nav>
          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-[#101827]/12 px-3 py-1.5 text-xs font-semibold text-[#566071] sm:inline">
              M1 Token Reducer
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  className="group relative flex size-9 cursor-pointer items-center justify-center rounded-full bg-[#101827] text-sm font-bold text-[#d8ff4f] shadow-sm transition-all hover:scale-105 hover:ring-4 hover:ring-[#101827]/15 focus:outline-hidden"
                  aria-label="Account Menu"
                >
                  M1
                  <span className="absolute -bottom-0.5 -right-0.5 size-3 rounded-full border-2 border-[#f4f5f0] bg-emerald-500" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-72 rounded-2xl border border-[#101827]/12 bg-white/95 p-2 shadow-2xl backdrop-blur-md"
                align="end"
                sideOffset={8}
              >
                <DropdownMenuLabel className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="grid size-10 place-items-center rounded-full bg-[#101827] text-sm font-black text-[#d8ff4f] shadow-inner">
                      M1
                    </div>
                    <div className="flex flex-col min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate text-sm font-bold text-[#101827]">
                          {userProfile.name}
                        </span>
                        <span className="rounded-full bg-[#6a7516]/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#6a7516]">
                          Demo
                        </span>
                      </div>
                      <span className="truncate text-xs text-[#566071]">
                        {userProfile.email}
                      </span>
                    </div>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator className="my-1 bg-[#101827]/8" />
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onClick={() => setActiveDialog("profile")}
                    className="flex cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-[#101827] transition-colors hover:bg-[#101827]/5"
                  >
                    <User size={16} className="text-[#566071]" />
                    <span>Profile & Account</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => setActiveDialog("api_keys")}
                    className="flex cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-[#101827] transition-colors hover:bg-[#101827]/5"
                  >
                    <Key size={16} className="text-[#566071]" />
                    <span>API Keys & Integration</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => setActiveDialog("security")}
                    className="flex cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-[#101827] transition-colors hover:bg-[#101827]/5"
                  >
                    <Shield size={16} className="text-[#566071]" />
                    <span>Security & Roles</span>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator className="my-1 bg-[#101827]/8" />
                <DropdownMenuItem
                  onClick={() => {
                    toast.success(
                      "Account menu synced. Active session: Demo User",
                    );
                  }}
                  className="flex cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50"
                >
                  <LogOut size={16} />
                  <span>Switch / Reset Session</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <section
        id="workspace"
        className="mx-auto flex h-[calc(100dvh-64px)] max-w-[1440px] flex-col overflow-hidden px-5 py-4 sm:px-8"
      >
        <div className="mb-3 flex min-h-14 items-center justify-between gap-4">
          <div>
            <p className="mb-1 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.14em] text-[#5a6655]">
              <Sparkles size={13} />{" "}
              {activeTool === "pdf"
                ? "PDF analyzer & token reducer"
                : "Prompt optimizer"}
            </p>
            <h1 className="text-2xl font-semibold leading-none tracking-[-0.045em] sm:text-3xl">
              {activeTool === "pdf" ? (
                <>
                  Less document.{" "}
                  <span className="text-[#6d7d17]">More signal.</span>
                </>
              ) : (
                <>
                  Sharper prompts.{" "}
                  <span className="text-[#6d7d17]">Fewer tokens.</span>
                </>
              )}
            </h1>
          </div>
          <div className="flex rounded-xl border border-[#101827]/10 bg-white p-1 shadow-sm">
            <button
              onClick={() => setActiveTool("pdf")}
              className={`h-9 rounded-lg px-3 text-xs font-semibold transition ${
                activeTool === "pdf"
                  ? "bg-[#101827] text-white"
                  : "text-[#68717e] hover:text-[#101827]"
              }`}
            >
              PDF Analyzer
            </button>
            <button
              onClick={() => setActiveTool("qa")}
              className={`relative h-9 rounded-lg px-3 text-xs font-semibold transition ${
                activeTool === "qa"
                  ? "bg-[#101827] text-white"
                  : "text-[#68717e] hover:text-[#101827]"
              }`}
            >
              Doc Q&A
              {indexStatus === "indexed" && (
                <span className="absolute -right-0.5 -top-0.5 size-2 rounded-full bg-[#6a7516] ring-2 ring-white" />
              )}
            </button>
            <button
              onClick={() => setActiveTool("prompt")}
              className={`h-9 rounded-lg px-3 text-xs font-semibold transition ${
                activeTool === "prompt"
                  ? "bg-[#101827] text-white"
                  : "text-[#68717e] hover:text-[#101827]"
              }`}
            >
              Prompt Chat
            </button>
          </div>
        </div>

        {/* PDF Analyzer Tab */}
        <div
          className={`${activeTool === "pdf" ? "grid" : "hidden"} min-h-0 flex-1 overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_28px_80px_rgba(16,24,39,0.09)] lg:grid-cols-[0.72fr_1.28fr]`}
        >
          <section className="overflow-y-auto border-b border-[#101827]/10 bg-[#101827] p-5 text-white lg:border-b-0 lg:border-r">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <span className="mb-1 block text-[11px] font-bold uppercase tracking-[0.16em] text-[#d8ff4f]">
                  01 / Source
                </span>
                <h2 className="text-xl font-semibold tracking-[-0.03em]">
                  Add your document
                </h2>
              </div>
              <span className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/55">
                PDF · max 30 MB
              </span>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              className="sr-only"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {!file ? (
              <button
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  chooseFile(event.dataTransfer.files?.[0]);
                }}
                className="group flex min-h-[250px] w-full flex-col items-center justify-center rounded-[18px] border border-dashed border-white/25 bg-white/[0.045] p-8 text-center transition hover:border-[#d8ff4f]/70 hover:bg-white/[0.07]"
              >
                <span className="mb-5 grid size-14 place-items-center rounded-full bg-[#d8ff4f] text-[#101827] transition-transform group-hover:-translate-y-1">
                  <UploadCloud size={24} strokeWidth={2.2} />
                </span>
                <strong className="text-lg">Drop a PDF here</strong>
                <span className="mt-2 max-w-[240px] text-sm leading-6 text-white/50">
                  or click to browse from your computer
                </span>
              </button>
            ) : (
              <div className="min-h-[250px] rounded-[18px] border border-white/15 bg-white/[0.055] p-5">
                <div className="flex items-start gap-4">
                  <span className="grid size-12 shrink-0 place-items-center rounded-xl bg-[#d8ff4f] text-[#101827]">
                    <FileText size={22} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-semibold">{file.name}</p>
                    <p className="mt-1 text-sm text-white/45">
                      {analysisResult
                        ? `${analysisResult.page_count} pages`
                        : "Ready to parse"}{" "}
                      · {file.size}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setFile(null);
                      setStatus("idle");
                      setAnalysisResult(null);
                      setErrorMessage(null);
                      resetDocumentState();
                      if (inputRef.current) inputRef.current.value = "";
                    }}
                    className="grid size-8 place-items-center rounded-full text-white/45 hover:bg-white/10 hover:text-white"
                    aria-label="Remove file"
                  >
                    <X size={17} />
                  </button>
                </div>
                <div className="mt-7 border-t border-white/10 pt-5">
                  <div className="mb-3 flex justify-between text-xs font-medium">
                    <span className="text-white/50">Extraction pipeline</span>
                    <span className="text-[#d8ff4f]">
                      {status === "working"
                        ? "Processing..."
                        : status === "done"
                          ? "Cleaned & Formatted"
                          : "Ready"}
                    </span>
                  </div>
                  <Progress
                    value={
                      status === "working" ? 64 : status === "done" ? 100 : 25
                    }
                    className="bg-white/10 [&_[data-slot=progress-indicator]]:bg-[#d8ff4f]"
                  />
                  <div className="mt-5 grid grid-cols-3 gap-2 text-center">
                    <Info
                      value={
                        analysisResult ? `${analysisResult.page_count}` : "—"
                      }
                      label="Pages"
                    />
                    <Info
                      value={
                        analysisResult ? `${analysisResult.table_count}` : "—"
                      }
                      label="Tables"
                    />
                    <Info
                      value={
                        analysisResult ? `${analysisResult.section_count}` : "—"
                      }
                      label="Sections"
                    />
                  </div>
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
                <AlertCircle size={16} className="shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold">Error processing document</p>
                  <p className="mt-0.5 text-white/70">{errorMessage}</p>
                </div>
              </div>
            )}

            <Button
              onClick={analyze}
              disabled={!file || status === "working"}
              className="mt-6 h-12 w-full rounded-xl bg-[#d8ff4f] font-bold text-[#101827] hover:bg-[#c9ef43] transition disabled:opacity-50"
            >
              {status === "working" ? (
                "Extracting & compressing..."
              ) : status === "done" ? (
                <>
                  <RotateCcw className="mr-2" size={16} /> Re-analyze PDF
                </>
              ) : (
                <>
                  Analyze PDF <ArrowRight className="ml-2" size={16} />
                </>
              )}
            </Button>
          </section>

          {/* Results Section */}
          <section className="flex min-h-0 flex-col bg-[#fbfbf8]">
            <div className="flex min-h-[80px] flex-col justify-between gap-3 border-b border-[#101827]/10 px-5 py-4 sm:flex-row sm:items-center sm:px-8">
              <div>
                <span className="block text-[11px] font-bold uppercase tracking-[0.16em] text-[#7c872f]">
                  02 / Result
                </span>
                <h2 className="mt-1 text-xl font-semibold tracking-[-0.03em]">
                  Optimized Markdown
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={copyMarkdown}
                  disabled={!analysisResult}
                  className="rounded-lg border-[#101827]/12 bg-transparent hover:bg-black/5"
                >
                  <Clipboard size={15} className="mr-1.5" />{" "}
                  {copied ? "Copied" : "Copy"}
                </Button>
                <Button
                  onClick={exportMarkdown}
                  disabled={!analysisResult}
                  className="rounded-lg bg-[#101827] text-white hover:bg-[#263247]"
                >
                  <Download size={15} className="mr-1.5" /> Export .md
                </Button>
              </div>
            </div>

            <div className="grid flex-1 overflow-hidden md:grid-cols-[1fr_200px]">
              <article className="min-h-0 overflow-y-auto border-b border-[#101827]/10 p-5 sm:p-6 md:border-b-0 md:border-r">
                {status === "idle" && !analysisResult ? (
                  <EmptyResult />
                ) : status === "working" ? (
                  <Working />
                ) : status === "error" ? (
                  <div className="flex h-full flex-col items-center justify-center p-6 text-center text-[#737b86]">
                    <AlertCircle size={32} className="mb-2 text-red-500" />
                    <p className="font-semibold text-[#101827]">
                      Extraction Error
                    </p>
                    <p className="mt-1 text-sm max-w-sm">{errorMessage}</p>
                    <Button
                      onClick={analyze}
                      variant="outline"
                      className="mt-4"
                    >
                      Try Again
                    </Button>
                  </div>
                ) : analysisResult?.is_scanned ||
                  (!analysisResult?.markdown && status === "done") ? (
                  <div className="flex h-full flex-col items-center justify-center p-6 text-center text-[#737b86]">
                    <AlertCircle size={32} className="mb-2 text-amber-500" />
                    <p className="font-semibold text-[#101827]">
                      No Text Extracted
                    </p>
                    <p className="mt-2 text-sm max-w-sm leading-6">
                      This PDF appears to be image-based or scanned. PyMuPDF
                      could not extract selectable text. Try a text-based PDF or
                      run it through an OCR tool first.
                    </p>
                  </div>
                ) : (
                  <div className="prose prose-sm max-w-none text-[#2d3644]">
                    <pre className="whitespace-pre-wrap font-mono text-[13px] leading-6 text-[#3d4655] select-text">
                      {analysisResult?.markdown}
                    </pre>
                  </div>
                )}
              </article>

              <aside className="bg-[#f5f5ef] p-5 sm:p-6 overflow-y-auto">
                <div className="mb-6">
                  <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-[#69717d]">
                    Token Score
                  </p>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-4xl font-semibold tracking-[-0.06em]">
                      {analysisResult
                        ? Math.round(analysisResult.reduction_percent)
                        : 0}
                    </span>
                    <span className="text-sm text-[#69717d]">% less</span>
                  </div>
                  <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-[#dfe2d7]">
                    <span
                      className="absolute inset-y-0 left-0 rounded-full bg-[#83951c] transition-all duration-500"
                      style={{
                        width: `${Math.min(100, Math.max(0, analysisResult?.reduction_percent || 0))}%`,
                      }}
                    />
                  </div>
                </div>

                <Stat
                  label="Original"
                  value={
                    analysisResult
                      ? analysisResult.original_tokens.toLocaleString()
                      : "—"
                  }
                  sub="tokens"
                />
                <Stat
                  label="Optimized"
                  value={
                    analysisResult
                      ? analysisResult.optimized_tokens.toLocaleString()
                      : "—"
                  }
                  sub="tokens"
                  accent
                />
                <Stat
                  label="Saved"
                  value={
                    analysisResult
                      ? analysisResult.tokens_saved.toLocaleString()
                      : "—"
                  }
                  sub="tokens"
                />

                <div className="mt-6 rounded-xl border border-[#101827]/10 bg-white p-3.5">
                  <p className="flex items-center gap-1.5 text-xs font-bold text-[#5e691e]">
                    <Check size={14} /> Structure preserved
                  </p>
                  <p className="mt-1.5 text-xs leading-5 text-[#69717d]">
                    Headings, tables, and bullet lists formatted for RAG
                    ingestion.
                  </p>
                </div>

                {analysisResult && (
                  <div className="mt-4 rounded-xl border border-[#101827]/10 bg-white p-3.5">
                    <p className="text-xs font-bold text-[#101827]">
                      Chunks generated:{" "}
                      <span className="text-[#5e691e]">
                        {analysisResult.chunk_count}
                      </span>
                    </p>
                    <p className="mt-1 text-[11px] leading-4 text-[#69717d]">
                      Semantic chunks ready for Member 2 Vector Store &
                      Retriever.
                    </p>
                  </div>
                )}
              </aside>
            </div>
          </section>
        </div>

        {/* Prompt Optimizer Tab */}
        <section
          className={`${activeTool === "prompt" ? "flex" : "hidden"} min-h-0 flex-1`}
          id="prompt-optimizer"
        >
          <div className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_20px_60px_rgba(16,24,39,0.07)]">
            <div className="flex h-16 items-center justify-between border-b border-[#101827]/10 px-5 sm:px-7">
              <div className="flex items-center gap-3">
                <span className="grid size-8 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                  <WandSparkles size={16} />
                </span>
                <div>
                  <p className="text-sm font-semibold">
                    Prompt chat & optimizer
                  </p>
                  <p className="text-xs text-[#7b8490]">
                    Refines intent, strips fluff, cuts token cost
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setPrompt("");
                  setPromptResult(null);
                  setPromptHistory([]);
                  setCopiedId(null);
                }}
                className="rounded-lg border border-[#101827]/10 px-3 py-2 text-xs font-semibold text-[#596270] hover:bg-[#f4f5f0] transition"
              >
                New prompt
              </button>
            </div>

            <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col px-5 py-5 sm:px-8">
              {promptHistory.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-6 text-center">
                  <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-[#101827] text-[#d8ff4f]">
                    <WandSparkles size={21} />
                  </span>
                  <h3 className="text-2xl font-semibold tracking-[-0.035em]">
                    How can I improve your prompt?
                  </h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-[#747d89]">
                    Paste a prompt below. TokenWise strips conversational
                    filler, refines directives, extracts constraints, and cuts
                    token cost.
                  </p>
                  <div className="mt-6 flex flex-wrap justify-center gap-2 max-w-xl">
                    {[
                      "Can you please write a python script to parse csv files?",
                      "Act as an expert in Python and explain how to optimize tokens because they cost money. Make sure to give a clear table.",
                      "Could you please summarize this clearly?",
                      "What is tokenization in LLMs?",
                    ].map((example, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => {
                          setPrompt(example);
                        }}
                        className="rounded-full border border-[#101827]/10 bg-white px-3 py-1.5 text-xs text-[#525b68] hover:border-[#6f7f16] hover:bg-[#f8f9f5] transition text-left"
                      >
                        ⚡{" "}
                        {example.length > 48
                          ? example.slice(0, 48) + "..."
                          : example}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="flex-1 space-y-6 overflow-y-auto pb-6 pr-1">
                  {promptHistory.map((item) => (
                    <div key={item.id} className="space-y-4">
                      {/* User prompt message */}
                      <div className="ml-auto max-w-[85%] rounded-[20px] rounded-br-md bg-[#eef0e7] px-4 py-3 text-sm leading-6 text-[#2d3644]">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-[#6a7516] mb-1">
                          Original prompt
                        </p>
                        <p className="whitespace-pre-wrap">{item.original}</p>
                      </div>

                      {/* Optimizer response */}
                      <div className="flex gap-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                          <WandSparkles size={15} />
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-sm font-semibold">
                              Tokenwise Optimized
                            </p>
                            <div className="flex flex-wrap items-center gap-1.5">
                              {item.result.method === "approx" && (
                                <span className="rounded-full bg-amber-100 px-2.5 py-1 text-[11px] font-semibold text-amber-800">
                                  ≈ estimate
                                </span>
                              )}
                              {item.result.semantic_warning && (
                                <span className="rounded-full bg-red-100 px-2.5 py-1 text-[11px] font-semibold text-red-800">
                                  Review meaning
                                </span>
                              )}
                              <span className="rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                                {item.result.tokens_saved > 0
                                  ? `~${item.result.reduction_percent}% fewer tokens`
                                  : item.result.optimized_tokens >
                                      item.result.original_tokens
                                    ? `${item.result.optimized_tokens - item.result.original_tokens} more tokens`
                                    : "Optimized directive"}{" "}
                                ({item.result.original_tokens} →{" "}
                                {item.result.optimized_tokens})
                              </span>
                            </div>
                          </div>
                          <div className="rounded-xl border border-[#101827]/10 bg-[#f8f9f5] p-4">
                            <pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-[#3d4654]">
                              {item.result.optimized}
                            </pre>
                          </div>
                          <div className="mt-3 flex flex-wrap items-center gap-3">
                            <button
                              onClick={() =>
                                copyOptimizedPrompt(
                                  item.id,
                                  item.result.optimized,
                                )
                              }
                              className="flex items-center gap-1.5 text-xs font-semibold text-[#626c78] hover:text-[#101827]"
                            >
                              <Clipboard size={14} />{" "}
                              {copiedId === item.id ? "Copied!" : "Copy prompt"}
                            </button>
                            {item.result.changes &&
                              item.result.changes.length > 0 && (
                                <span className="text-[11px] text-[#788290]">
                                  • {item.result.changes.join(" • ")}
                                </span>
                              )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="rounded-[24px] border border-[#101827]/15 bg-[#f8f8f5] p-3 shadow-[0_12px_35px_rgba(16,24,39,0.1)] focus-within:border-[#899a20]/60">
                <Textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      handleOptimizePrompt();
                    }
                  }}
                  placeholder="Paste your prompt here to reduce tokens and improve clarity..."
                  className="min-h-[76px] resize-none border-0 bg-transparent px-2 py-1 text-base leading-6 shadow-none focus-visible:ring-0"
                />
                <div className="mt-2 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#9198a1] px-2">
                      {prompt.length} chars · ≈
                      {prompt.trim()
                        ? Math.max(1, Math.ceil(prompt.trim().length / 4))
                        : 0}{" "}
                      tokens
                    </span>
                  </div>
                  <button
                    onClick={() => handleOptimizePrompt()}
                    disabled={!prompt.trim() || promptLoading}
                    className="grid size-9 place-items-center rounded-full bg-[#101827] text-white transition hover:bg-[#6f7f16] disabled:bg-[#c7cbc4]"
                    aria-label="Send prompt"
                  >
                    <ArrowUp size={18} strokeWidth={2.6} />
                  </button>
                </div>
              </div>
              <p className="mt-3 text-center text-[11px] text-[#949aa2]">
                Press Enter to optimize · Shift + Enter for a new line
              </p>
            </div>
          </div>
        </section>

        {/* ── M2: Document Q&A Tab ────────────────────────────────────────── */}
        <section
          className={`${activeTool === "qa" ? "flex" : "hidden"} min-h-0 flex-1`}
          id="document-qa"
        >
          <div className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_20px_60px_rgba(16,24,39,0.07)]">
            {/* Header */}
            <div className="flex h-16 items-center justify-between border-b border-[#101827]/10 px-5 sm:px-7">
              <div className="flex items-center gap-3">
                <span className="grid size-8 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                  <Search size={16} />
                </span>
                <div>
                  <p className="text-sm font-semibold">Document Q&A</p>
                  <p className="text-xs text-[#7b8490]">
                    Ask questions about your uploaded PDF — powered by RAG
                    retrieval
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {indexStatus === "indexed" && documentId && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                    <span className="size-1.5 rounded-full bg-[#6a7516]" />{" "}
                    Indexed: {documentId}
                  </span>
                )}
                {indexStatus === "indexing" && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-1 text-[11px] font-semibold text-blue-600">
                    <Loader2 size={12} className="animate-spin" /> Indexing…
                  </span>
                )}
                <button
                  onClick={() => {
                    setQueryText("");
                    setQueryResult(null);
                    setQueryHistory([]);
                  }}
                  className="rounded-lg border border-[#101827]/10 px-3 py-2 text-xs font-semibold text-[#596270] hover:bg-[#f4f5f0] transition"
                >
                  Clear history
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col px-5 py-5 sm:px-8">
              {queryError && indexStatus !== "error" && (
                <p
                  role="alert"
                  className="mb-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
                >
                  {queryError}
                </p>
              )}
              {/* Empty state or conversation */}
              {queryHistory.length === 0 && !queryResult ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-10 text-center">
                  <span className="mb-5 grid size-12 place-items-center rounded-2xl bg-[#101827] text-[#d8ff4f]">
                    <Search size={21} />
                  </span>
                  <h3 className="text-2xl font-semibold tracking-[-0.035em]">
                    Ask your document anything
                  </h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-[#747d89]">
                    {indexStatus === "indexed"
                      ? "Your document is indexed and ready. Type a question below to retrieve the most relevant passages."
                      : indexStatus === "indexing"
                        ? "Indexing your document chunks... This will only take a moment."
                        : indexStatus === "error"
                          ? queryError ||
                            "Document indexing failed. Please try analyzing the PDF again."
                          : "Upload and analyze a PDF first. After analysis, the document chunks will be automatically indexed for semantic search."}
                  </p>
                  {(indexStatus === "idle" || indexStatus === "error") && (
                    <Button
                      onClick={() => setActiveTool("pdf")}
                      className="mt-5 rounded-lg bg-[#101827] text-white hover:bg-[#263247]"
                    >
                      Go to PDF Analyzer{" "}
                      <ArrowRight className="ml-2" size={14} />
                    </Button>
                  )}
                </div>
              ) : (
                <div className="flex-1 space-y-4 overflow-y-auto pb-6">
                  {queryHistory.map((entry, i) => (
                    <div key={i} className="space-y-3">
                      {/* User question */}
                      <div className="ml-auto max-w-[85%] rounded-[20px] rounded-br-md bg-[#eef0e7] px-4 py-3 text-sm leading-6 text-[#2d3644]">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-[#6a7516] mb-1">
                          Question
                        </p>
                        {entry.query}
                      </div>

                      {/* RAG results */}
                      <div className="flex gap-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                          <Search size={15} />
                        </span>
                        <div className="min-w-0 flex-1">
                          {/* Direct Answer */}
                          {entry.result.direct_answer && (
                            <div className="mb-3 rounded-2xl border border-[#6a7516]/30 bg-[#f5f8eb] p-4 text-[#101827] shadow-[0_4px_16px_rgba(106,117,22,0.06)]">
                              <div className="flex items-center gap-2 mb-1.5">
                                <span className="grid size-5 place-items-center rounded-full bg-[#6a7516] text-[#d8ff4f]">
                                  <Sparkles size={11} />
                                </span>
                                <span className="text-[11px] font-bold uppercase tracking-wider text-[#6a7516]">
                                  Direct Answer
                                </span>
                              </div>
                              <p className="text-sm font-semibold leading-relaxed text-[#1e2714]">
                                {entry.result.direct_answer}
                              </p>
                            </div>
                          )}

                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-xs font-bold uppercase tracking-wider text-[#7b8490]">
                              Source Evidence & Chunks
                            </p>
                            <div className="flex items-center gap-2">
                              <span className="rounded-full bg-[#f0f1eb] px-2.5 py-1 text-[11px] font-semibold text-[#596270]">
                                {entry.result.query_analysis.intent} ·{" "}
                                {entry.result.query_analysis.complexity}
                              </span>
                              <span className="rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                                {entry.result.num_chunks_retrieved} chunks
                              </span>
                            </div>
                          </div>

                          {/* Keywords */}
                          {entry.result.query_analysis.keywords.length > 0 && (
                            <div className="mb-3 flex flex-wrap gap-1.5">
                              {entry.result.query_analysis.keywords.map(
                                (kw, ki) => (
                                  <span
                                    key={ki}
                                    className="rounded-md bg-[#f4f5f0] px-2 py-0.5 text-[11px] font-medium text-[#596270]"
                                  >
                                    {kw}
                                  </span>
                                ),
                              )}
                            </div>
                          )}

                          {/* Chunk results */}
                          <div className="space-y-2">
                            {entry.result.retrieved_chunks.map((chunk, ci) => (
                              <div
                                key={ci}
                                className="rounded-xl border border-[#101827]/10 bg-[#f8f9f5] p-4 transition hover:border-[#899a20]/40"
                              >
                                <div className="mb-2 flex items-center justify-between">
                                  <span className="text-[11px] font-bold text-[#5e691e]">
                                    § {chunk.metadata.section}
                                  </span>
                                  <div className="flex items-center gap-2">
                                    <span className="text-[11px] text-[#868d96]">
                                      {chunk.metadata.tokens} tokens
                                    </span>
                                    <span
                                      className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
                                        chunk.relevance_score >= 0.7
                                          ? "bg-green-100 text-green-700"
                                          : chunk.relevance_score >= 0.4
                                            ? "bg-amber-100 text-amber-700"
                                            : "bg-gray-100 text-gray-600"
                                      }`}
                                    >
                                      {(chunk.relevance_score * 100).toFixed(1)}
                                      % match
                                    </span>
                                  </div>
                                </div>
                                <p className="text-sm leading-6 text-[#3d4654]">
                                  {chunk.text}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Query input */}
              <div className="rounded-[24px] border border-[#101827]/15 bg-[#f8f8f5] p-3 shadow-[0_12px_35px_rgba(16,24,39,0.1)] focus-within:border-[#899a20]/60">
                <Textarea
                  value={queryText}
                  onChange={(event) => setQueryText(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      handleQuery();
                    }
                  }}
                  placeholder={
                    indexStatus === "indexed"
                      ? "Ask a question about your document..."
                      : "Index a document first to enable Q&A..."
                  }
                  disabled={indexStatus !== "indexed"}
                  className="min-h-[56px] resize-none border-0 bg-transparent px-2 py-1 text-base leading-6 shadow-none focus-visible:ring-0 disabled:opacity-50"
                />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-[#9198a1] px-2">
                    {indexStatus === "indexed"
                      ? "Semantic search powered by M2 RAG"
                      : "Waiting for document index…"}
                  </span>
                  <button
                    onClick={handleQuery}
                    disabled={
                      !queryText.trim() ||
                      queryLoading ||
                      indexStatus !== "indexed"
                    }
                    className="grid size-9 place-items-center rounded-full bg-[#101827] text-white transition hover:bg-[#6f7f16] disabled:bg-[#c7cbc4]"
                    aria-label="Search document"
                  >
                    {queryLoading ? (
                      <Loader2 size={18} className="animate-spin" />
                    ) : (
                      <Search size={18} strokeWidth={2.6} />
                    )}
                  </button>
                </div>
              </div>
              <p className="mt-3 text-center text-[11px] text-[#949aa2]">
                Press Enter to search · Results ranked by semantic similarity
              </p>
            </div>
          </div>
        </section>
      </section>

      {/* How It Works Dialog */}
      <Dialog
        open={activeDialog === "how"}
        onOpenChange={(open) => !open && setActiveDialog(null)}
      >
        <DialogContent className="sm:max-w-lg rounded-2xl p-6 bg-white border border-[#101827]/12 shadow-2xl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-[#101827]">
              How Tokenwise works
            </DialogTitle>
            <DialogDescription className="text-sm text-[#566071]">
              Turn long documents and rough prompts into more focused,
              token-efficient content.
            </DialogDescription>
          </DialogHeader>

          <ol className="space-y-4 py-3 text-sm text-[#566071]">
            <li className="flex gap-3">
              <span className="grid size-7 shrink-0 place-items-center rounded-full bg-[#101827] text-xs font-bold text-[#d8ff4f]">
                1
              </span>
              <span>
                <strong className="text-[#101827]">Analyze a PDF.</strong>{" "}
                Upload a document to extract its text, clean repeated content,
                and create optimized Markdown.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="grid size-7 shrink-0 place-items-center rounded-full bg-[#101827] text-xs font-bold text-[#d8ff4f]">
                2
              </span>
              <span>
                <strong className="text-[#101827]">Ask questions.</strong> The
                analyzed document is indexed so you can search its most relevant
                passages in Doc Q&amp;A.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="grid size-7 shrink-0 place-items-center rounded-full bg-[#101827] text-xs font-bold text-[#d8ff4f]">
                3
              </span>
              <span>
                <strong className="text-[#101827]">Refine prompts.</strong> Use
                Prompt Chat to make instructions clearer and more concise.
              </span>
            </li>
          </ol>

          <div className="flex justify-end">
            <Button
              onClick={() => {
                setActiveDialog(null);
                setActiveTool("pdf");
              }}
              className="rounded-xl bg-[#101827] text-white hover:bg-[#101827]/90"
            >
              Open workspace
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Account Profile Dialog */}
      <Dialog
        open={activeDialog === "profile"}
        onOpenChange={(open) => !open && setActiveDialog(null)}
      >
        <DialogContent className="sm:max-w-md rounded-2xl p-6 bg-white border border-[#101827]/12 shadow-2xl">
          <DialogHeader>
            <div className="flex items-center gap-3 mb-2">
              <div className="grid size-12 place-items-center rounded-full bg-[#101827] text-lg font-black text-[#d8ff4f]">
                M1
              </div>
              <div>
                <DialogTitle className="text-xl font-bold text-[#101827]">
                  Account Profile
                </DialogTitle>
                <DialogDescription className="text-sm text-[#566071]">
                  Manage your personal details and workspace permissions.
                </DialogDescription>
              </div>
            </div>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-[#566071] mb-1 block">
                Full Name
              </label>
              <input
                type="text"
                value={userProfile.name}
                onChange={(e) =>
                  setUserProfile((p) => ({ ...p, name: e.target.value }))
                }
                className="w-full rounded-xl border border-[#101827]/15 bg-[#f4f5f0]/50 px-3.5 py-2 text-sm font-medium text-[#101827] focus:outline-hidden focus:ring-2 focus:ring-[#101827]"
              />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-[#566071] mb-1 block">
                Email Address
              </label>
              <input
                type="email"
                value={userProfile.email}
                onChange={(e) =>
                  setUserProfile((p) => ({ ...p, email: e.target.value }))
                }
                className="w-full rounded-xl border border-[#101827]/15 bg-[#f4f5f0]/50 px-3.5 py-2 text-sm font-medium text-[#101827] focus:outline-hidden focus:ring-2 focus:ring-[#101827]"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-[#566071] mb-1 block">
                  Role
                </label>
                <input
                  type="text"
                  value={userProfile.role}
                  readOnly
                  className="w-full rounded-xl border border-[#101827]/10 bg-[#f4f5f0] px-3.5 py-2 text-sm font-medium text-[#566071] cursor-not-allowed"
                />
              </div>
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-[#566071] mb-1 block">
                  Organization
                </label>
                <input
                  type="text"
                  value={userProfile.organization}
                  readOnly
                  className="w-full rounded-xl border border-[#101827]/10 bg-[#f4f5f0] px-3.5 py-2 text-sm font-medium text-[#566071] cursor-not-allowed"
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-4 pt-3 border-t border-[#101827]/10">
            <Button
              variant="outline"
              onClick={() => setActiveDialog(null)}
              className="rounded-xl"
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                toast.success("Profile changes saved successfully!");
                setActiveDialog(null);
              }}
              className="rounded-xl bg-[#101827] text-white hover:bg-[#101827]/90"
            >
              Save Changes
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* API Keys Dialog */}
      <Dialog
        open={activeDialog === "api_keys"}
        onOpenChange={(open) => !open && setActiveDialog(null)}
      >
        <DialogContent className="sm:max-w-lg rounded-2xl p-6 bg-white border border-[#101827]/12 shadow-2xl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-[#101827] flex items-center gap-2">
              <Key size={20} className="text-[#6a7516]" /> API Keys & Developer
              Access
            </DialogTitle>
            <DialogDescription className="text-sm text-[#566071]">
              Use secret keys to access the TokenWise PDF reduction & RAG REST
              API programmatically.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-3">
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <p className="font-semibold">Demo workspace</p>
              <p className="mt-1 text-xs leading-5">
                No API key is configured or generated in this demo.
              </p>
            </div>

            <div className="space-y-2 text-xs text-[#566071]">
              <div className="flex justify-between border-b border-[#101827]/8 py-1.5">
                <span>Backend Endpoint</span>
                <span className="font-mono text-[#101827] font-semibold">
                  {API_BASE ? `${API_BASE}/api` : "Same-origin /api"}
                </span>
              </div>
              <div className="flex justify-between border-b border-[#101827]/8 py-1.5">
                <span>Rate Limit</span>
                <span className="font-semibold text-[#101827]">
                  Not configured
                </span>
              </div>
            </div>
          </div>

          <div className="flex justify-end mt-2">
            <Button
              onClick={() => setActiveDialog(null)}
              className="rounded-xl bg-[#101827] text-white"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Security & Roles Dialog */}
      <Dialog
        open={activeDialog === "security"}
        onOpenChange={(open) => !open && setActiveDialog(null)}
      >
        <DialogContent className="sm:max-w-md rounded-2xl p-6 bg-white border border-[#101827]/12 shadow-2xl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-[#101827] flex items-center gap-2">
              <Shield size={20} className="text-[#6a7516]" /> Security &
              Permissions
            </DialogTitle>
            <DialogDescription className="text-sm text-[#566071]">
              Workspace security configuration and access details.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3 py-3 text-sm">
            <div className="flex items-center justify-between p-3 rounded-xl bg-[#f4f5f0] border border-[#101827]/8">
              <div className="flex items-center gap-2">
                <Shield size={16} className="text-emerald-600" />
                <span className="font-semibold text-[#101827]">
                  2-Factor Authentication
                </span>
              </div>
              <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-bold">
                Demo · not configured
              </span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-[#f4f5f0] border border-[#101827]/8">
              <div className="flex items-center gap-2">
                <Activity size={16} className="text-[#566071]" />
                <span className="font-semibold text-[#101827]">
                  Session Address
                </span>
              </div>
              <span className="text-xs text-[#566071] font-mono">
                Not available in demo
              </span>
            </div>
          </div>

          <div className="flex justify-end mt-2">
            <Button
              onClick={() => setActiveDialog(null)}
              className="rounded-xl bg-[#101827] text-white"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Toaster position="top-right" />
    </main>
  );
}

function Info({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg bg-white/[0.055] px-2 py-3">
      <strong className="block text-base">{value}</strong>
      <span className="text-[11px] text-white/40">{label}</span>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div className="border-t border-[#101827]/10 py-3.5">
      <p className="text-xs text-[#737b86]">{label}</p>
      <p
        className={`mt-1 text-xl font-semibold tracking-[-0.03em] ${accent ? "text-[#718015]" : ""}`}
      >
        {value}{" "}
        <span className="text-xs font-normal text-[#868d96]">{sub}</span>
      </p>
    </div>
  );
}

function EmptyResult() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <span className="mb-4 grid size-12 place-items-center rounded-xl border border-[#101827]/10 bg-white text-[#76808e]">
        <FileText size={21} />
      </span>
      <h3 className="font-semibold">Your Markdown will appear here</h3>
      <p className="mt-2 max-w-xs text-sm leading-6 text-[#79818c]">
        Upload a PDF to extract text, clean headers/footers, and reduce token
        usage.
      </p>
    </div>
  );
}

function Working() {
  return (
    <div className="flex h-full flex-col items-center justify-center">
      <div className="mb-5 size-9 animate-spin rounded-full border-[3px] border-[#dfe4c0] border-t-[#778616]" />
      <p className="font-semibold">Extracting signal & structuring Markdown…</p>
      <p className="mt-2 text-sm text-[#79818c]">
        Removing repeating headers, formatting tables, and cutting tokens
      </p>
    </div>
  );
}
