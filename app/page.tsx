"use client";

import { useRef, useState } from "react";
import { ArrowRight, ArrowUp, Check, ChevronDown, Clipboard, Download, FileText, Menu, Paperclip, Plus, RotateCcw, Sparkles, UploadCloud, WandSparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

const sampleMarkdown = `# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the **Transformer**, based solely on attention mechanisms.

## 1. Introduction

Recurrent neural networks have been firmly established as state-of-the-art approaches in sequence modeling and transduction problems.

### Key contributions

- Introduces a model based entirely on attention
- Removes recurrence and convolutions
- Enables significantly more parallelization
- Reaches a new state of the art in translation quality

## 2. Model architecture

The Transformer follows an encoder–decoder structure. Each encoder layer has two sub-layers: multi-head self-attention and a position-wise feed-forward network.`;

type FileDetails = { name: string; size: string; pages: number };

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<FileDetails | null>(null);
  const [level, setLevel] = useState("Balanced");
  const [status, setStatus] = useState<"idle" | "working" | "done">("idle");
  const [copied, setCopied] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [optimizedPrompt, setOptimizedPrompt] = useState("");
  const [activeTool, setActiveTool] = useState<"pdf" | "prompt">("pdf");

  const chooseFile = (selected?: File) => {
    if (!selected) return;
    setFile({ name: selected.name, size: `${(selected.size / 1024 / 1024).toFixed(1)} MB`, pages: Math.max(8, Math.min(120, Math.round(selected.size / 39000))) });
    setStatus("idle");
  };

  const analyze = () => {
    if (!file) return;
    setStatus("working");
    window.setTimeout(() => setStatus("done"), 1200);
  };

  const copyMarkdown = async () => {
    await navigator.clipboard?.writeText(sampleMarkdown);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const optimizePrompt = () => {
    if (!prompt.trim()) return;
    setOptimizedPrompt(`Task: ${prompt.trim()}

Requirements:
- Return a clear, accurate answer.
- Use concise language and a structured format.
- Include only information relevant to the requested outcome.
- State any necessary assumptions briefly.`);
  };

  return (
    <main className="h-screen overflow-hidden bg-[#f4f5f0] text-[#101827]">
      <header className="border-b border-[#101827]/10 bg-[#f4f5f0]/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-5 sm:px-8">
          <a href="#" className="flex items-center gap-3" aria-label="Tokenwise home">
            <span className="grid size-9 place-items-center rounded-[10px] bg-[#101827] text-[#d8ff4f]"><span className="text-lg font-black tracking-[-0.16em]">T/</span></span>
            <span className="text-[19px] font-bold tracking-[-0.04em]">Tokenwise</span>
          </a>
          <nav className="hidden items-center gap-8 text-sm font-medium text-[#566071] md:flex" aria-label="Main navigation">
            <a className="text-[#101827]" href="#workspace">Workspace</a>
            <a className="transition-colors hover:text-[#101827]" href="#how">How it works</a>
            <a className="transition-colors hover:text-[#101827]" href="#history">History</a>
          </nav>
          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-[#101827]/12 px-3 py-1.5 text-xs font-semibold text-[#566071] sm:inline">0 / 5 files today</span>
            <button className="grid size-9 place-items-center rounded-full border border-[#101827]/12 bg-white md:hidden" aria-label="Open menu"><Menu size={18} /></button>
            <div className="hidden size-9 place-items-center rounded-full bg-[#101827] text-sm font-bold text-white md:grid">A</div>
          </div>
        </div>
      </header>

      <section id="workspace" className="mx-auto flex h-[calc(100vh-64px)] max-w-[1440px] flex-col overflow-hidden px-5 py-4 sm:px-8">
        <div className="mb-3 flex min-h-14 items-center justify-between gap-4">
          <div>
            <p className="mb-1 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.14em] text-[#5a6655]"><Sparkles size={13} /> {activeTool === "pdf" ? "PDF analyzer" : "Prompt optimizer"}</p>
            <h1 className="text-2xl font-semibold leading-none tracking-[-0.045em] sm:text-3xl">{activeTool === "pdf" ? <>Less document. <span className="text-[#6d7d17]">More signal.</span></> : <>Sharper prompts. <span className="text-[#6d7d17]">Fewer tokens.</span></>}</h1>
          </div>
          <div className="flex rounded-xl border border-[#101827]/10 bg-white p-1 shadow-sm"><button onClick={() => setActiveTool("pdf")} className={`h-9 rounded-lg px-3 text-xs font-semibold ${activeTool === "pdf" ? "bg-[#101827] text-white" : "text-[#68717e]"}`}>PDF Analyzer</button><button onClick={() => setActiveTool("prompt")} className={`h-9 rounded-lg px-3 text-xs font-semibold ${activeTool === "prompt" ? "bg-[#101827] text-white" : "text-[#68717e]"}`}>Prompt Chat</button></div>
        </div>

        <div className={`${activeTool === "pdf" ? "grid" : "hidden"} min-h-0 flex-1 overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_28px_80px_rgba(16,24,39,0.09)] lg:grid-cols-[0.72fr_1.28fr]`}>
          <section className="overflow-hidden border-b border-[#101827]/10 bg-[#101827] p-5 text-white lg:border-b-0 lg:border-r">
            <div className="mb-6 flex items-center justify-between">
              <div><span className="mb-1 block text-[11px] font-bold uppercase tracking-[0.16em] text-[#d8ff4f]">01 / Source</span><h2 className="text-xl font-semibold tracking-[-0.03em]">Add your document</h2></div>
              <span className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/55">PDF · max 25 MB</span>
            </div>

            <input ref={inputRef} type="file" accept="application/pdf" className="sr-only" onChange={(event) => chooseFile(event.target.files?.[0])} />
            {!file ? (
              <button onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files?.[0]); }} className="group flex min-h-[270px] w-full flex-col items-center justify-center rounded-[18px] border border-dashed border-white/25 bg-white/[0.045] p-8 text-center transition hover:border-[#d8ff4f]/70 hover:bg-white/[0.07]">
                <span className="mb-5 grid size-14 place-items-center rounded-full bg-[#d8ff4f] text-[#101827] transition-transform group-hover:-translate-y-1"><UploadCloud size={24} strokeWidth={2.2} /></span>
                <strong className="text-lg">Drop a PDF here</strong><span className="mt-2 max-w-[240px] text-sm leading-6 text-white/50">or click to browse from your computer</span>
              </button>
            ) : (
              <div className="min-h-[270px] rounded-[18px] border border-white/15 bg-white/[0.055] p-5">
                <div className="flex items-start gap-4">
                  <span className="grid size-12 shrink-0 place-items-center rounded-xl bg-[#d8ff4f] text-[#101827]"><FileText size={22} /></span>
                  <div className="min-w-0 flex-1"><p className="truncate font-semibold">{file.name}</p><p className="mt-1 text-sm text-white/45">{file.pages} pages · {file.size}</p></div>
                  <button onClick={() => { setFile(null); setStatus("idle"); }} className="grid size-8 place-items-center rounded-full text-white/45 hover:bg-white/10 hover:text-white" aria-label="Remove file"><X size={17} /></button>
                </div>
                <div className="mt-7 border-t border-white/10 pt-5">
                  <div className="mb-3 flex justify-between text-xs font-medium"><span className="text-white/50">Document quality</span><span className="text-[#d8ff4f]">Text recognized</span></div>
                  <Progress value={status === "working" ? 64 : 100} className="bg-white/10 [&_[data-slot=progress-indicator]]:bg-[#d8ff4f]" />
                  <div className="mt-5 grid grid-cols-3 gap-2 text-center"><Info value={`${file.pages}`} label="Pages" /><Info value="3" label="Tables" /><Info value="12" label="Sections" /></div>
                </div>
              </div>
            )}

            <div className="mt-6">
              <label className="mb-3 block text-xs font-semibold uppercase tracking-[0.12em] text-white/45">Compression</label>
              <Tabs value={level} onValueChange={setLevel}><TabsList className="grid h-11 w-full grid-cols-3 bg-white/[0.07] p-1">
                {["Light", "Balanced", "Maximum"].map((item) => <TabsTrigger key={item} value={item} className="text-xs text-white/50 data-[state=active]:bg-white data-[state=active]:text-[#101827]">{item}</TabsTrigger>)}
              </TabsList></Tabs>
              <p className="mt-3 text-xs leading-5 text-white/40">{level === "Light" ? "Preserves most detail and formatting." : level === "Maximum" ? "Keeps only the most essential information." : "Best balance of context and token savings."}</p>
            </div>
            <Button onClick={analyze} disabled={!file || status === "working"} className="mt-6 h-12 w-full rounded-xl bg-[#d8ff4f] font-bold text-[#101827] hover:bg-[#c9ef43]">
              {status === "working" ? "Analyzing document…" : status === "done" ? <><RotateCcw /> Analyze again</> : <>Analyze PDF <ArrowRight /></>}
            </Button>
          </section>

          <section className="flex min-h-0 flex-col bg-[#fbfbf8]">
            <div className="flex min-h-[88px] flex-col justify-between gap-4 border-b border-[#101827]/10 px-5 py-4 sm:flex-row sm:items-center sm:px-8">
              <div><span className="block text-[11px] font-bold uppercase tracking-[0.16em] text-[#7c872f]">02 / Result</span><h2 className="mt-1 text-xl font-semibold tracking-[-0.03em]">Optimized Markdown</h2></div>
              <div className="flex items-center gap-2"><Button variant="outline" onClick={copyMarkdown} className="rounded-lg border-[#101827]/12 bg-transparent"><Clipboard /> {copied ? "Copied" : "Copy"}</Button><Button className="rounded-lg bg-[#101827] text-white hover:bg-[#263247]"><Download /> Export .md</Button></div>
            </div>
            <div className="grid flex-1 md:grid-cols-[1fr_170px]">
              <article className="min-h-0 overflow-auto border-b border-[#101827]/10 p-5 sm:p-6 md:border-b-0 md:border-r">
                {status === "idle" && !file ? <EmptyResult /> : status === "working" ? <Working /> : <pre className="whitespace-pre-wrap font-mono text-[13px] leading-6 text-[#3d4655]">{sampleMarkdown}</pre>}
              </article>
              <aside className="bg-[#f5f5ef] p-5 sm:p-6">
                <div className="mb-6 flex items-center justify-between md:block"><div><p className="text-[11px] font-bold uppercase tracking-[0.14em] text-[#69717d]">Token score</p><div className="mt-2 flex items-baseline gap-1"><span className="text-4xl font-semibold tracking-[-0.06em]">68</span><span className="text-sm text-[#69717d]">% less</span></div></div><div className="relative mt-5 hidden h-2 overflow-hidden rounded-full bg-[#dfe2d7] md:block"><span className="absolute inset-y-0 left-0 w-[68%] rounded-full bg-[#83951c]" /></div></div>
                <Stat label="Original" value="8,420" sub="tokens" /><Stat label="Optimized" value="2,694" sub="tokens" accent /><Stat label="Saved" value="5,726" sub="tokens" />
                <div className="mt-7 rounded-xl border border-[#101827]/10 bg-white p-4"><p className="flex items-center gap-2 text-xs font-bold text-[#5e691e]"><Check size={14} /> Structure preserved</p><p className="mt-2 text-xs leading-5 text-[#69717d]">Headings, lists, emphasis, and tables are retained.</p></div>
              </aside>
            </div>
          </section>
        </div>
        <div id="how" className="hidden"><span>Your files are processed securely and deleted after conversion.</span><button>Supported document types <ChevronDown size={14} /></button></div>

        <section className={`${activeTool === "prompt" ? "flex" : "hidden"} min-h-0 flex-1`} id="prompt-optimizer">
          <div className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-[24px] border border-[#101827]/10 bg-white shadow-[0_20px_60px_rgba(16,24,39,0.07)]">
            <div className="flex h-16 items-center justify-between border-b border-[#101827]/10 px-5 sm:px-7">
              <div className="flex items-center gap-3"><span className="grid size-8 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]"><WandSparkles size={16} /></span><div><p className="text-sm font-semibold">Prompt chat</p><p className="text-xs text-[#7b8490]">Balanced optimization</p></div></div>
              <button onClick={() => { setPrompt(""); setOptimizedPrompt(""); }} className="rounded-lg border border-[#101827]/10 px-3 py-2 text-xs font-semibold text-[#596270] hover:bg-[#f4f5f0]">New chat</button>
            </div>

            <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col px-5 py-5 sm:px-8">
              {!optimizedPrompt ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-10 text-center"><span className="mb-5 grid size-12 place-items-center rounded-2xl bg-[#101827] text-[#d8ff4f]"><WandSparkles size={21} /></span><h3 className="text-2xl font-semibold tracking-[-0.035em]">How can I improve your prompt?</h3><p className="mt-2 max-w-md text-sm leading-6 text-[#747d89]">Paste a rough prompt below. I’ll make the goal clearer, remove repetition, and keep the important context.</p></div>
              ) : (
                <div className="flex-1 space-y-7 pb-10">
                  <div className="ml-auto max-w-[82%] rounded-[20px] rounded-br-md bg-[#eef0e7] px-4 py-3 text-sm leading-6 text-[#2d3644]">{prompt}</div>
                  <div className="flex gap-3"><span className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#101827] text-[#d8ff4f]"><WandSparkles size={15} /></span><div className="min-w-0 flex-1"><div className="mb-3 flex items-center justify-between"><p className="text-sm font-semibold">Tokenwise</p><span className="rounded-full bg-[#edf4c9] px-2.5 py-1 text-[11px] font-semibold text-[#647111]">~32% shorter</span></div><pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-[#3d4654]">{optimizedPrompt}</pre><button onClick={() => navigator.clipboard?.writeText(optimizedPrompt)} className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-[#626c78] hover:text-[#101827]"><Clipboard size={14} /> Copy</button></div></div>
                </div>
              )}

              <div className="rounded-[24px] border border-[#101827]/15 bg-[#f8f8f5] p-3 shadow-[0_12px_35px_rgba(16,24,39,0.1)] focus-within:border-[#899a20]/60">
                <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); optimizePrompt(); } }} placeholder="Ask Tokenwise to improve your prompt" className="min-h-[76px] resize-none border-0 bg-transparent px-2 py-1 text-base leading-6 shadow-none focus-visible:ring-0" />
                <div className="mt-2 flex items-center justify-between">
                  <div className="flex items-center gap-2"><button className="grid size-9 place-items-center rounded-full border border-[#101827]/10 bg-white text-[#4e5866]" aria-label="Add attachment"><Plus size={18} /></button><button className="flex h-9 items-center gap-2 rounded-full border border-[#101827]/10 bg-white px-3 text-xs font-semibold text-[#4e5866]"><Paperclip size={14} /> Add PDF</button><span className="hidden text-xs text-[#9198a1] sm:inline">{prompt.length} characters</span></div>
                  <button onClick={optimizePrompt} disabled={!prompt.trim()} className="grid size-9 place-items-center rounded-full bg-[#101827] text-white transition hover:bg-[#6f7f16] disabled:bg-[#c7cbc4]" aria-label="Send prompt"><ArrowUp size={18} strokeWidth={2.6} /></button>
                </div>
              </div>
              <p className="mt-3 text-center text-[11px] text-[#949aa2]">Press Enter to send · Shift + Enter for a new line</p>
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}

function Info({ value, label }: { value: string; label: string }) { return <div className="rounded-lg bg-white/[0.055] px-2 py-3"><strong className="block text-base">{value}</strong><span className="text-[11px] text-white/40">{label}</span></div>; }
function Stat({ label, value, sub, accent = false }: { label: string; value: string; sub: string; accent?: boolean }) { return <div className="border-t border-[#101827]/10 py-4"><p className="text-xs text-[#737b86]">{label}</p><p className={`mt-1 text-xl font-semibold tracking-[-0.03em] ${accent ? "text-[#718015]" : ""}`}>{value} <span className="text-xs font-normal text-[#868d96]">{sub}</span></p></div>; }
function EmptyResult() { return <div className="flex h-full flex-col items-center justify-center text-center"><span className="mb-4 grid size-12 place-items-center rounded-xl border border-[#101827]/10 bg-white text-[#76808e]"><FileText size={21} /></span><h3 className="font-semibold">Your Markdown will appear here</h3><p className="mt-2 max-w-xs text-sm leading-6 text-[#79818c]">Add a PDF and choose how much you want to reduce it.</p></div>; }
function Working() { return <div className="flex h-full flex-col items-center justify-center"><div className="mb-5 size-9 animate-spin rounded-full border-[3px] border-[#dfe4c0] border-t-[#778616]" /><p className="font-semibold">Finding the signal…</p><p className="mt-2 text-sm text-[#79818c]">Cleaning structure and removing repetition</p></div>; }
