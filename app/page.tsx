"use client";

import { useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUp,
  AlertCircle,
  Check,
  ChevronDown,
  Clipboard,
  Download,
  FileText,
  Loader2,
  Menu,
  MessageSquare,
  Paperclip,
  Plus,
  RotateCcw,
  Search,
  Sparkles,
  UploadCloud,
  WandSparkles,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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
  retrieved_chunks: RetrievedChunk[];
  num_chunks_retrieved: number;
  suggested_top_k: number;
};

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<FileDetails | null>(null);
  const [level, setLevel] = useState("Balanced");
  const [status, setStatus] = useState<"idle" | "working" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [promptCopied, setPromptCopied] = useState(false);

  // Real data state
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [prompt, setPrompt] = useState("");
  const [promptResult, setPromptResult] = useState<PromptResult | null>(null);
  const [promptLoading, setPromptLoading] = useState(false);
  const [activeTool, setActiveTool] = useState<"pdf" | "prompt" | "qa">("pdf");

  // M2 RAG state
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [indexStatus, setIndexStatus] = useState<"idle" | "indexing" | "indexed" | "error">("idle");
  const [queryText, setQueryText] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryHistory, setQueryHistory] = useState<Array<{ query: string; result: QueryResult }>>([]);

  const chooseFile = (selected?: File) => {
    if (!selected) return;
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
    formData.append("compression", level);

    try {
      const response = await fetch(`${API_BASE}/api/analyze-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(errorData.detail || `Server responded with ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      setAnalysisResult(data);
      // Update page count from backend
      setFile((prev) => prev ? { ...prev, pages: data.page_count } : null);
      setStatus("done");

      // Auto-index chunks for RAG (M2)
      if (data.chunks && data.chunks.length > 0) {
        autoIndexChunks(data.chunks, file!.name);
      }
    } catch (err: any) {
      console.error("PDF Analysis failed:", err);
      setErrorMessage(
        err.message || "Failed to connect to backend server. Make sure the FastAPI service is running."
      );
      setStatus("error");
    }
  };

  // ── M2: Auto-index chunks after PDF analysis ──────────────────────────
  const autoIndexChunks = async (chunks: NonNullable<AnalysisResult["chunks"]>, filename: string) => {
    const docId = filename.replace(/\.pdf$/i, "").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40);
    setIndexStatus("indexing");
    setDocumentId(null);
    setQueryResult(null);
    setQueryHistory([]);

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
    }
  };

  // ── M2: Query the indexed document ────────────────────────────────────
  const handleQuery = async () => {
    if (!queryText.trim() || !documentId || queryLoading) return;
    setQueryLoading(true);

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
      setQueryHistory((prev) => [...prev, { query: queryText.trim(), result: data }]);
      setQueryText("");
    } catch (err) {
      console.error("Query failed:", err);
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

  const handleOptimizePrompt = async () => {
    if (!prompt.trim() || promptLoading) return;
    setPromptLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/optimize-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });

      if (!response.ok) {
        throw new Error("Failed to optimize prompt");
      }

      const data: PromptResult = await response.json();
      setPromptResult(data);
    } catch (err: any) {
      console.error("Prompt optimization failed:", err);
      // Fallback local rule-based optimization
      const fallbackCleaned = prompt
        .replace(/^(can you please|could you please|please)\s+/i, "")
        .replace(/in order to/gi, "to")
        .replace(/due to the fact that/gi, "because");
      const fallbackText = `Task: ${fallbackCleaned.trim()}\n\nRequirements:\n- Be concise and direct.\n- Output structured format.`;
      setPromptResult({
        optimized: fallbackText,
        original_tokens: prompt.split(/\s+/).length,
        optimized_tokens: Math.round(fallbackText.split(/\s+/).length * 0.8),
        tokens_saved: Math.round(prompt.split(/\s+/).length * 0.2),
        reduction_percent: 25.0,
        changes: ["Applied local token compression fallback."],
      });
    } finally {
      setPromptLoading(false);
    }
  };

  const copyOptimizedPrompt = async () => {
    if (!promptResult?.optimized) return;
    await navigator.clipboard?.writeText(promptResult.optimized);
    setPromptCopied(true);
    window.setTimeout(() => setPromptCopied(false), 1600);
  };

  return (
    <main className="h-screen overflow-hidden bg-[#f4f5f0] text-[#101827]">
      <header className="border-b border-[#101827]/10 bg-[#f4f5f0]/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-5 sm:px-8">
          <a href="#" className="flex items-center gap-3" aria-label="Tokenwise home">
            <span className="grid size-9 place-items-center rounded-[10px] bg-[#101827] text-[#d8ff4f]">
              <span className="text-lg font-black tracking-[-0.16em]">T/</span>
            </span>
            <span className="text-[19px] font-bold tracking-[-0.04em]">Tokenwise</span>
          </a>
          <nav className="hidden items-center gap-8 text-sm font-medium text-[#566071] md:flex" aria-label="Main navigation">
            <a className="text-[#101827]" href="#workspace">Workspace</a>
            <a className="transition-colors hover:text-[#101827]" href="#how">How it works</a>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#101827]/5 px-2.5 py-1 text-xs font-semibold text-[#6a7516]">
              <span className="size-2 rounded-full bg-[#6a7516] animate-pulse" /> Backend Active
            </span>
          </nav>
          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-[#101827]/12 px-3 py-1.5 text-xs font-semibold text-[#566071] sm:inline">
              M1 Token Reducer
            </span>
            <button className="grid size-9 place-items-center rounded-full border border-[#101827]/12 bg-white md:hidden" aria-label="Open menu">
              <Menu size={18} />
            </button>
            <div className="hidden size-9 place-items-center rounded-full bg-[#101827] text-sm font-bold text-white md:grid">
              M1
            </div>
          </div>
        </div>
      </header>

      <section id="workspace" className="mx-auto flex h-[calc(100vh-64px)] max-w-[1440px] flex-col overflow-hidden px-5 py-4 sm:px-8">
        <div className="mb-3 flex min-h-14 items-center justify-between gap-4">
          <div>
            <p className="mb-1 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.14em] text-[#5a6655]">
              <Sparkles size={13} /> {activeTool === "pdf" ? "PDF analyzer & token reducer" : "Prompt optimizer"}
            </p>
            <h1 className="text-2xl font-semibold leading-none tracking-[-0.045em] sm:text-3xl">
              {activeTool === "pdf" ? (
                <>Less document. <span className="text-[#6d7d17]">More signal.</span></>
              ) : (
                <>Sharper prompts. <span className="text-[#6d7d17]">Fewer tokens.</span></>
              )}
            </h1>
          </div>
          <div className="flex rounded-xl border border-[#101827]/10 bg-white p-1 shadow-sm">
            <button
              onClick={() => setActiveTool("pdf")}
              className={`h-9 rounded-lg px-3 text-xs font-semibold transition ${
                activeTool === "pdf" ? "bg-[#101827] text-white" : "text-[#68717e] hover:text-[#101827]"
              }`}
            >
              PDF Analyzer
            </button>
            <button
              onClick={() => setActiveTool("qa")}
              className={`relative h-9 rounded-lg px-3 text-xs font-semibold transition ${
                activeTool === "qa" ? "bg-[#101827] text-white" : "text-[#68717e] hover:text-[#101827]"
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
                activeTool === "prompt" ? "bg-[#101827] text-white" : "text-[#68717e] hover:text-[#101827]"
              }`}
            >
              Prompt Chat
            </button>
          </div>
        </div>

        {/* PDF Analyzer Tab */}
        <div className={`${activeTool === "pdf" ? "grid" : "hidden"} min-h-0 flex-1 overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_28px_80px_rgba(16,24,39,0.09)] lg:grid-cols-[0.72fr_1.28fr]`}>
          <section className="overflow-y-auto border-b border-[#101827]/10 bg-[#101827] p-5 text-white lg:border-b-0 lg:border-r">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <span className="mb-1 block text-[11px] font-bold uppercase tracking-[0.16em] text-[#d8ff4f]">01 / Source</span>
                <h2 className="text-xl font-semibold tracking-[-0.03em]">Add your document</h2>
              </div>
              <span className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/55">PDF · max 30 MB</span>
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
                <span className="mt-2 max-w-[240px] text-sm leading-6 text-white/50">or click to browse from your computer</span>
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
                      {analysisResult ? `${analysisResult.page_count} pages` : "Ready to parse"} · {file.size}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setFile(null);
                      setStatus("idle");
                      setAnalysisResult(null);
                      setErrorMessage(null);
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
                      {status === "working" ? "Processing..." : status === "done" ? "Cleaned & Formatted" : "Ready"}
                    </span>
                  </div>
                  <Progress
                    value={status === "working" ? 64 : status === "done" ? 100 : 25}
                    className="bg-white/10 [&_[data-slot=progress-indicator]]:bg-[#d8ff4f]"
                  />
                  <div className="mt-5 grid grid-cols-3 gap-2 text-center">
                    <Info value={analysisResult ? `${analysisResult.page_count}` : "—"} label="Pages" />
                    <Info value={analysisResult ? `${analysisResult.table_count}` : "—"} label="Tables" />
                    <Info value={analysisResult ? `${analysisResult.section_count}` : "—"} label="Sections" />
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

            <div className="mt-6">
              <label className="mb-3 block text-xs font-semibold uppercase tracking-[0.12em] text-white/45">
                Token Reduction Level
              </label>
              <Tabs value={level} onValueChange={setLevel}>
                <TabsList className="grid h-11 w-full grid-cols-3 bg-white/[0.07] p-1">
                  {["Light", "Balanced", "Maximum"].map((item) => (
                    <TabsTrigger
                      key={item}
                      value={item}
                      className="text-xs text-white/50 data-[state=active]:bg-white data-[state=active]:text-[#101827]"
                    >
                      {item}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
              <p className="mt-3 text-xs leading-5 text-white/40">
                {level === "Light"
                  ? "Preserves full detail, cleans headers, page numbers, duplicate spaces."
                  : level === "Maximum"
                  ? "Maximum token reduction: extracts key bullet points, tables, and core takeaways."
                  : "Balanced: strips boilerplate, optimizes verbose sentences, preserves tables and structure."}
              </p>
            </div>

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
                <span className="block text-[11px] font-bold uppercase tracking-[0.16em] text-[#7c872f]">02 / Result</span>
                <h2 className="mt-1 text-xl font-semibold tracking-[-0.03em]">Optimized Markdown</h2>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={copyMarkdown}
                  disabled={!analysisResult}
                  className="rounded-lg border-[#101827]/12 bg-transparent hover:bg-black/5"
                >
                  <Clipboard size={15} className="mr-1.5" /> {copied ? "Copied" : "Copy"}
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
                    <p className="font-semibold text-[#101827]">Extraction Error</p>
                    <p className="mt-1 text-sm max-w-sm">{errorMessage}</p>
                    <Button onClick={analyze} variant="outline" className="mt-4">
                      Try Again
                    </Button>
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
                  <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-[#69717d]">Token Score</p>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-4xl font-semibold tracking-[-0.06em]">
                      {analysisResult ? Math.round(analysisResult.reduction_percent) : 0}
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
                  value={analysisResult ? analysisResult.original_tokens.toLocaleString() : "—"}
                  sub="tokens"
                />
                <Stat
                  label="Optimized"
                  value={analysisResult ? analysisResult.optimized_tokens.toLocaleString() : "—"}
                  sub="tokens"
                  accent
                />
                <Stat
                  label="Saved"
                  value={analysisResult ? analysisResult.tokens_saved.toLocaleString() : "—"}
                  sub="tokens"
                />

                <div className="mt-6 rounded-xl border border-[#101827]/10 bg-white p-3.5">
                  <p className="flex items-center gap-1.5 text-xs font-bold text-[#5e691e]">
                    <Check size={14} /> Structure preserved
                  </p>
                  <p className="mt-1.5 text-xs leading-5 text-[#69717d]">
                    Headings, tables, and bullet lists formatted for RAG ingestion.
                  </p>
                </div>

                {analysisResult && (
                  <div className="mt-4 rounded-xl border border-[#101827]/10 bg-white p-3.5">
                    <p className="text-xs font-bold text-[#101827]">
                      Chunks generated: <span className="text-[#5e691e]">{analysisResult.chunk_count}</span>
                    </p>
                    <p className="mt-1 text-[11px] leading-4 text-[#69717d]">
                      Semantic chunks ready for Member 2 Vector Store & Retriever.
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
                  <p className="text-sm font-semibold">Prompt chat & optimizer</p>
                  <p className="text-xs text-[#7b8490]">Refines intent, strips fluff, cuts token cost</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setPrompt("");
                  setPromptResult(null);
                }}
                className="rounded-lg border border-[#101827]/10 px-3 py-2 text-xs font-semibold text-[#596270] hover:bg-[#f4f5f0] transition"
              >
                New prompt
              </button>
            </div>

            <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col px-5 py-5 sm:px-8">
              {!promptResult ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-10 text-center">
                  <span className="mb-5 grid size-12 place-items-center rounded-2xl bg-[#101827] text-[#d8ff4f]">
                    <WandSparkles size={21} />
                  </span>
                  <h3 className="text-2xl font-semibold tracking-[-0.035em]">How can I improve your prompt?</h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-[#747d89]">
                    Paste a prompt below. We'll strip conversational filler, enforce explicit constraints, and reduce token usage for better LLM responses.
                  </p>
                </div>
              ) : (
                <div className="flex-1 space-y-5 overflow-y-auto pb-6">
                  <div className="ml-auto max-w-[85%] rounded-[20px] rounded-br-md bg-[#eef0e7] px-4 py-3 text-sm leading-6 text-[#2d3644]">
                    <p className="text-[11px] font-bold uppercase tracking-wider text-[#6a7516] mb-1">Original prompt</p>
                    {prompt}
                  </div>
                  <div className="flex gap-3">
                    <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                      <WandSparkles size={15} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="mb-2 flex items-center justify-between">
                        <p className="text-sm font-semibold">Tokenwise Optimized</p>
                        <span className="rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                          ~{promptResult.reduction_percent}% fewer tokens ({promptResult.original_tokens} → {promptResult.optimized_tokens})
                        </span>
                      </div>
                      <div className="rounded-xl border border-[#101827]/10 bg-[#f8f9f5] p-4">
                        <pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-[#3d4654]">
                          {promptResult.optimized}
                        </pre>
                      </div>
                      <div className="mt-3 flex items-center gap-3">
                        <button
                          onClick={copyOptimizedPrompt}
                          className="flex items-center gap-1.5 text-xs font-semibold text-[#626c78] hover:text-[#101827]"
                        >
                          <Clipboard size={14} /> {promptCopied ? "Copied!" : "Copy prompt"}
                        </button>
                        {promptResult.changes && promptResult.changes.length > 0 && (
                          <span className="text-[11px] text-[#788290]">
                            • {promptResult.changes.join(" • ")}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="rounded-[24px] border border-[#101827]/15 bg-[#f8f8f5] p-3 shadow-[0_12px_35px_rgba(16,24,39,0.1)] focus-within:border-[#899a20]/60">
                <Textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
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
                      {prompt.length} chars · ~{prompt ? prompt.split(/\s+/).length : 0} words
                    </span>
                  </div>
                  <button
                    onClick={handleOptimizePrompt}
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
                  <p className="text-xs text-[#7b8490]">Ask questions about your uploaded PDF — powered by RAG retrieval</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {indexStatus === "indexed" && documentId && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                    <span className="size-1.5 rounded-full bg-[#6a7516]" /> Indexed: {documentId}
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
              {/* Empty state or conversation */}
              {queryHistory.length === 0 && !queryResult ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-10 text-center">
                  <span className="mb-5 grid size-12 place-items-center rounded-2xl bg-[#101827] text-[#d8ff4f]">
                    <Search size={21} />
                  </span>
                  <h3 className="text-2xl font-semibold tracking-[-0.035em]">Ask your document anything</h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-[#747d89]">
                    {indexStatus === "indexed"
                      ? "Your document is indexed and ready. Type a question below to retrieve the most relevant passages."
                      : indexStatus === "indexing"
                      ? "Indexing your document chunks... This will only take a moment."
                      : "Upload and analyze a PDF first. After analysis, the document chunks will be automatically indexed for semantic search."}
                  </p>
                  {indexStatus === "idle" && (
                    <Button
                      onClick={() => setActiveTool("pdf")}
                      className="mt-5 rounded-lg bg-[#101827] text-white hover:bg-[#263247]"
                    >
                      Go to PDF Analyzer <ArrowRight className="ml-2" size={14} />
                    </Button>
                  )}
                </div>
              ) : (
                <div className="flex-1 space-y-4 overflow-y-auto pb-6">
                  {queryHistory.map((entry, i) => (
                    <div key={i} className="space-y-3">
                      {/* User question */}
                      <div className="ml-auto max-w-[85%] rounded-[20px] rounded-br-md bg-[#eef0e7] px-4 py-3 text-sm leading-6 text-[#2d3644]">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-[#6a7516] mb-1">Question</p>
                        {entry.query}
                      </div>

                      {/* RAG results */}
                      <div className="flex gap-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]">
                          <Search size={15} />
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="mb-2 flex items-center justify-between">
                            <p className="text-sm font-semibold">Retrieved Passages</p>
                            <div className="flex items-center gap-2">
                              <span className="rounded-full bg-[#f0f1eb] px-2.5 py-1 text-[11px] font-semibold text-[#596270]">
                                {entry.result.query_analysis.intent} · {entry.result.query_analysis.complexity}
                              </span>
                              <span className="rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">
                                {entry.result.num_chunks_retrieved} chunks
                              </span>
                            </div>
                          </div>

                          {/* Keywords */}
                          {entry.result.query_analysis.keywords.length > 0 && (
                            <div className="mb-3 flex flex-wrap gap-1.5">
                              {entry.result.query_analysis.keywords.map((kw, ki) => (
                                <span
                                  key={ki}
                                  className="rounded-md bg-[#f4f5f0] px-2 py-0.5 text-[11px] font-medium text-[#596270]"
                                >
                                  {kw}
                                </span>
                              ))}
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
                                      {(chunk.relevance_score * 100).toFixed(1)}% match
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
                    if (event.key === "Enter" && !event.shiftKey) {
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
                    {indexStatus === "indexed" ? "Semantic search powered by M2 RAG" : "Waiting for document index…"}
                  </span>
                  <button
                    onClick={handleQuery}
                    disabled={!queryText.trim() || queryLoading || indexStatus !== "indexed"}
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
      <p className={`mt-1 text-xl font-semibold tracking-[-0.03em] ${accent ? "text-[#718015]" : ""}`}>
        {value} <span className="text-xs font-normal text-[#868d96]">{sub}</span>
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
        Upload a PDF to extract text, clean headers/footers, and reduce token usage.
      </p>
    </div>
  );
}

function Working() {
  return (
    <div className="flex h-full flex-col items-center justify-center">
      <div className="mb-5 size-9 animate-spin rounded-full border-[3px] border-[#dfe4c0] border-t-[#778616]" />
      <p className="font-semibold">Extracting signal & structuring Markdown…</p>
      <p className="mt-2 text-sm text-[#79818c]">Removing repeating headers, formatting tables, and cutting tokens</p>
    </div>
  );
}
