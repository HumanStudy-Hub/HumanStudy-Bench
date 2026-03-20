import { promises as fs } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import Link from "next/link";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";

const DOCS_DIR = path.join(process.cwd(), "..", "docs");

const docsMeta: Record<
  string,
  { title: string; description: string; order: number }
> = {
  what_to_submit: {
    title: "What Should I Submit?",
    description: "The big picture — what goes in a study folder",
    order: 1,
  },
  extract_from_paper: {
    title: "How to Extract Data from a Paper",
    description: "Turn a PDF into structured, replicable data",
    order: 2,
  },
  build_study_files: {
    title: "How to Build Your Study Files",
    description: "Schemas, code, and contracts for every file",
    order: 3,
  },
  submit_study: {
    title: "How to Submit Your Study",
    description: "Fork, verify, push, and ship it",
    order: 4,
  },
};

const CALLOUT_ICONS = {
  tip: '<svg class="callout-svg" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" /></svg>',
  info: '<svg class="callout-svg" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" /></svg>',
  warning: '<svg class="callout-svg" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" /></svg>',
  note: '<svg class="callout-svg" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" /></svg>',
};

function stripAdmonitions(md: string): string {
  return md.replace(
    /^!!! (\w+)(?: "([^"]*)")?\n((?:    .*\n?)*)/gm,
    (_match, type, title, body) => {
      const content = body.replace(/^    /gm, "").trim();
      const normalizedType = type === "note" ? "info" : type;
      const label = title || type.charAt(0).toUpperCase() + type.slice(1);
      const icon =
        CALLOUT_ICONS[normalizedType as keyof typeof CALLOUT_ICONS] ||
        CALLOUT_ICONS.info;
      
      return `\n<div class="callout callout-${normalizedType}">\n<span class="callout-icon" aria-hidden="true">${icon}</span>\n<div class="callout-body">\n<strong>${label}</strong>\n\n${content}\n\n</div>\n</div>\n\n`;
    }
  );
}

function fixInternalLinks(md: string): string {
  return md.replace(
    /\[([^\]]+)\]\((?!https?:\/\/)([a-z_]+)\.md(?:#([a-z_-]+))?\)/g,
    (_match, text, slug, anchor) => {
      const href = anchor ? `/docs/${slug}#${anchor}` : `/docs/${slug}`;
      return `[${text}](${href})`;
    }
  );
}

export async function generateStaticParams() {
  return Object.keys(docsMeta).map((slug) => ({ slug }));
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const meta = docsMeta[slug];
  if (!meta) notFound();

  let content = "";
  try {
    const raw = await fs.readFile(path.join(DOCS_DIR, `${slug}.md`), "utf8");
    const withoutH1 = raw.replace(/^# .+\n/, "");
    content = fixInternalLinks(stripAdmonitions(withoutH1));
  } catch {
    notFound();
  }

  const allDocs = Object.entries(docsMeta).sort((a, b) => a[1].order - b[1].order);
  const currentIdx = allDocs.findIndex(([s]) => s === slug);
  const prev = currentIdx > 0 ? allDocs[currentIdx - 1] : null;
  const next =
    currentIdx < allDocs.length - 1 ? allDocs[currentIdx + 1] : null;

  return (
    <div className="bg-white min-h-screen">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex">
          {/* Sidebar — GitHub Docs style */}
          <aside className="hidden lg:block w-64 shrink-0 border-r border-gray-200 pr-6 py-8">
            <div className="sticky top-8">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">
                Contributor Tutorial
              </p>
              <nav className="space-y-0">
                {allDocs.map(([s, m]) => {
                  const isActive = s === slug;
                  return (
                    <Link
                      key={s}
                      href={`/docs/${s}`}
                      className={`block py-1.5 pl-4 -ml-px text-sm border-l-2 transition-colors ${
                        isActive
                          ? "border-cyan-600 text-gray-900 font-semibold"
                          : "border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300"
                      }`}
                    >
                      {m.title}
                    </Link>
                  );
                })}
              </nav>
            </div>
          </aside>

          {/* Main content */}
          <main className="min-w-0 flex-1 py-8 lg:pl-10">
            <div className="max-w-3xl">
              {/* Breadcrumb + title + description (inside content column) */}
              <div className="mb-8 border-b border-gray-200 pb-8">
                <p className="text-sm text-gray-500 mb-3 font-medium">
                  <Link
                    href="/contribute#documentation"
                    className="hover:text-cyan-600 hover:underline transition-colors"
                  >
                    Contribute
                  </Link>
                  {" / "}
                  <span className="text-gray-900">Docs</span>
                </p>
                <h1 className="text-4xl font-bold font-serif text-gray-900 mt-1 mb-4 leading-tight tracking-tight">
                  {meta.title}
                </h1>
                <p className="text-xl text-gray-600 font-light">
                  {meta.description}
                </p>
              </div>

              {/* Mobile nav — horizontal scroll */}
              <div className="lg:hidden mb-6 -mx-4 overflow-x-auto">
                <div className="flex gap-2 px-4 pb-2 min-w-max">
                  {allDocs.map(([s, m]) => (
                    <Link
                      key={s}
                      href={`/docs/${s}`}
                      className={`shrink-0 rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                        s === slug
                          ? "border-cyan-600 bg-cyan-50 text-cyan-700"
                          : "border-gray-200 text-gray-600 hover:border-gray-300 hover:text-gray-900"
                      }`}
                    >
                      {m.title}
                    </Link>
                  ))}
                </div>
              </div>

              {/* Markdown content — GitHub Docs typography */}
              <article
                className="
                  prose prose-gray max-w-none font-serif
                  prose-p:text-[#1F2328] prose-p:leading-[1.7]
                  prose-h2:font-serif prose-h2:text-[1.5rem] prose-h2:font-bold prose-h2:pb-2 prose-h2:border-b prose-h2:border-gray-200 prose-h2:mt-[24px] prose-h2:mb-[16px] prose-h2:text-[#1F2328]
                  prose-h3:font-serif prose-h3:text-[1.25rem] prose-h3:font-bold prose-h3:mt-[24px] prose-h3:mb-[16px] prose-h3:text-[#1F2328]
                  prose-li:text-[#1F2328] prose-li:leading-[1.7]
                  prose-strong:text-[#1F2328] prose-strong:font-bold
                  prose-a:text-[#0969da] prose-a:no-underline hover:prose-a:underline
                  prose-code:before:content-none prose-code:after:content-none
                  prose-code:bg-[#eff1f3] prose-code:px-[0.4em] prose-code:py-[0.2em] prose-code:rounded-md prose-code:text-[85%] prose-code:text-[#1F2328] prose-code:font-mono
                  prose-pre:bg-[#f6f8fa] prose-pre:rounded-md prose-pre:text-[85%] prose-pre:border prose-pre:border-[#d0d7de] prose-pre:text-[#1F2328] prose-pre:p-4 prose-pre:font-mono
                  prose-thead:bg-[#f6f8fa] prose-th:text-left prose-th:font-semibold prose-th:text-[#1F2328] prose-th:text-sm prose-th:border prose-th:border-[#d0d7de] prose-th:px-4 prose-th:py-2 prose-th:font-sans
                  prose-td:border prose-td:border-[#d0d7de] prose-td:px-4 prose-td:py-2 prose-td:text-[#1F2328] prose-td:font-sans
                  prose-table:text-sm prose-table:border-collapse
                  prose-hr:border-[#d0d7de] prose-hr:my-6 prose-hr:h-[0.25em] prose-hr:bg-[#d0d7de]
                  prose-blockquote:border-l-[0.25em] prose-blockquote:border-[#d0d7de] prose-blockquote:pl-4 prose-blockquote:text-[#656d76] prose-blockquote:not-italic prose-blockquote:bg-transparent prose-blockquote:py-0
                  prose-img:rounded-md
                "
              >
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight, rehypeRaw]}
                >
                  {content}
                </Markdown>
              </article>

              {/* Prev/Next — flat GitHub Docs style */}
              <div className="mt-12 pt-8 border-t border-gray-200 grid grid-cols-2 gap-4">
                {prev ? (
                  <Link
                    href={`/docs/${prev[0]}`}
                    className="flex items-center gap-3 rounded-lg border border-gray-200 px-4 py-3 hover:border-[#0969da] hover:text-[#0969da] transition-colors"
                  >
                    <svg
                      className="w-4 h-4 shrink-0 text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={2}
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M15.75 19.5L8.25 12l7.5-7.5"
                      />
                    </svg>
                    <div className="min-w-0">
                      <p className="text-xs text-gray-500">Previous</p>
                      <p className="text-sm font-semibold truncate mt-0.5">
                        {prev[1].title}
                      </p>
                    </div>
                  </Link>
                ) : (
                  <div />
                )}
                {next ? (
                  <Link
                    href={`/docs/${next[0]}`}
                    className="flex items-center justify-end gap-3 rounded-lg border border-gray-200 px-4 py-3 hover:border-[#0969da] hover:text-[#0969da] transition-colors text-right"
                  >
                    <div className="min-w-0">
                      <p className="text-xs text-gray-500">Next</p>
                      <p className="text-sm font-semibold truncate mt-0.5">
                        {next[1].title}
                      </p>
                    </div>
                    <svg
                      className="w-4 h-4 shrink-0 text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={2}
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M8.25 4.5l7.5 7.5-7.5 7.5"
                      />
                    </svg>
                  </Link>
                ) : (
                  <div />
                )}
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
