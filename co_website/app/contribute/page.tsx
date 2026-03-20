import { promises as fs } from "fs";
import path from "path";
import Link from "next/link";
import ContributeUploadForm from "@/components/ContributeUploadForm";

type Contributor = { name?: string; github?: string };
type StudySummary = {
  study_id: string;
  title: string;
  authors: string[];
  year: number | null;
  contributors?: Contributor[];
};

export default async function ContributePage() {
  const filePath = path.join(process.cwd(), "data/studies_index.json");
  let studies: StudySummary[] = [];
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const data = JSON.parse(raw);
    studies = data.studies ?? [];
  } catch (e) {
    console.error("Could not read studies index", e);
  }

  const foundational = studies.slice(0, 12);
  const community = studies.slice(12);

  return (
    <div className="bg-white min-h-screen">
      {/* Hero */}
      <div className="border-t border-gray-100 bg-white">
        <div className="mx-auto max-w-5xl px-6 lg:px-8 py-16 sm:py-20 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">
            Contribute
          </h1>
          <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
            Add a new study to HumanStudy-Bench.
          </p>
          <div className="mt-6 mx-auto max-w-2xl rounded-xl border border-cyan-100 bg-cyan-50/40 px-6 py-5 text-sm leading-relaxed text-gray-700 space-y-1.5">
            <p>HumanStudy-Bench is a non-profit and open-source platform.</p>
            <p>We welcome collaborators from all backgrounds!</p>
            <p>All accepted contributions will be acknowledged and credited.</p>
          </div>
          {/* Section nav */}
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <a href="#studies" className="rounded-full border border-gray-200 px-4 py-1.5 text-sm font-medium text-gray-700 hover:border-cyan-300 hover:text-cyan-700 transition-colors">
              Studies
            </a>
            <a href="#how-to-contribute" className="rounded-full border border-gray-200 px-4 py-1.5 text-sm font-medium text-gray-700 hover:border-cyan-300 hover:text-cyan-700 transition-colors">
              How to Contribute
            </a>
            <Link
              href="/docs/what_to_submit"
              className="group relative rounded-full px-5 py-1.5 text-sm font-semibold text-cyan-700 transition-all hover:text-cyan-800"
            >
              <span className="absolute inset-0 rounded-full bg-cyan-400/20 animate-pulse" />
              <span className="absolute inset-0 rounded-full border-2 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.4)]" />
              <span className="relative flex items-center gap-1.5">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
                Step-by-step Tutorial &rarr;
              </span>
            </Link>
          </div>
        </div>
      </div>

      {/* Studies Catalog */}
      <section id="studies" className="scroll-mt-20 border-t border-gray-100 py-12">
        <div className="mx-auto max-w-5xl px-6 lg:px-8">
          <h2 className="text-2xl font-bold tracking-tight text-gray-900 font-serif mb-8">
            Study Catalog
          </h2>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {foundational.map((study, idx) => (
              <div key={study.study_id} className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-3 py-2.5 hover:border-cyan-300 hover:shadow-sm transition-all">
                <div className="flex items-center gap-3 overflow-hidden">
                  <span className="inline-flex h-6 min-w-8 items-center justify-center rounded-md border border-cyan-100 bg-cyan-50 px-1 text-[10px] font-bold text-cyan-700 shrink-0">
                    {(idx + 1).toString().padStart(3, "0")}
                  </span>
                  <Link
                    href={`/studies/${study.study_id}`}
                    className="text-xs font-semibold text-gray-900 hover:text-cyan-700 transition-colors truncate"
                  >
                    {study.title}
                  </Link>
                </div>
                {study.contributors?.[0]?.github && (
                  <a
                    href={study.contributors[0].github.startsWith("http") ? study.contributors[0].github : `https://github.com/${study.contributors[0].github}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] text-gray-500 hover:text-cyan-700 transition-colors shrink-0 ml-4"
                  >
                    @{study.contributors[0].github.replace(/^https?:\/\/github\.com\//, "")}
                  </a>
                )}
              </div>
            ))}
          </div>

          {community.length > 0 && (
            <>
              <div className="my-8 flex items-center gap-4">
                <div className="flex-1 border-t border-gray-300" />
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Community Contributions
                </span>
                <div className="flex-1 border-t border-gray-300" />
              </div>

              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {community.map((study, idx) => (
                  <div key={study.study_id} className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-3 py-2.5 hover:border-cyan-300 hover:shadow-sm transition-all">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <span className="inline-flex h-6 min-w-8 items-center justify-center rounded-md border border-cyan-100 bg-cyan-50 px-1 text-[10px] font-bold text-cyan-700 shrink-0">
                        {(idx + 13).toString().padStart(3, "0")}
                      </span>
                      <Link
                        href={`/studies/${study.study_id}`}
                        className="text-xs font-semibold text-gray-900 hover:text-cyan-700 transition-colors truncate"
                      >
                        {study.title}
                      </Link>
                    </div>
                    {study.contributors?.[0]?.github && (
                      <a
                        href={study.contributors[0].github.startsWith("http") ? study.contributors[0].github : `https://github.com/${study.contributors[0].github}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-gray-500 hover:text-cyan-700 transition-colors shrink-0 ml-4"
                      >
                        @{study.contributors[0].github.replace(/^https?:\/\/github\.com\//, "")}
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </section>

      {/* How to Contribute */}
      <section id="how-to-contribute" className="scroll-mt-20 border-t border-gray-100 py-12">
        <div className="mx-auto max-w-5xl px-6 lg:px-8">
          <h2 className="text-2xl font-bold tracking-tight text-gray-900 font-serif mb-4">
            How to Contribute
          </h2>
          <p className="text-gray-600 mb-8">
            There are two ways to submit a study. We recommend contributing via GitHub.
          </p>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* GitHub PR path */}
            <div className="rounded-2xl border-2 border-cyan-200 bg-cyan-50/30 p-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="inline-flex items-center rounded-md bg-cyan-600 px-2 py-0.5 text-[10px] font-bold text-white uppercase tracking-wide">
                  Recommended
                </span>
                <h3 className="text-lg font-semibold text-gray-900">GitHub Pull Request</h3>
              </div>
              <ol className="space-y-3 text-sm text-gray-700">
                <li className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-white text-xs font-bold">1</span>
                  <span>Fork and clone the <a href="https://github.com/HumanStudy-Hub/HumanStudy-Bench" target="_blank" rel="noopener noreferrer" className="text-cyan-600 hover:underline">repository</a></span>
                </li>
                <li className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-white text-xs font-bold">2</span>
                  <span>Create a branch (e.g. <code className="bg-white px-1 rounded text-xs">contrib-yourgithubid-013</code>)</span>
                </li>
                <li className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-white text-xs font-bold">3</span>
                  <span>Add your study under <code className="bg-white px-1 rounded text-xs">studies/</code> with <code className="bg-white px-1 rounded text-xs">index.json</code>, <code className="bg-white px-1 rounded text-xs">source/</code>, <code className="bg-white px-1 rounded text-xs">scripts/</code>, and <code className="bg-white px-1 rounded text-xs">README.md</code></span>
                </li>
                <li className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-white text-xs font-bold">4</span>
                  <span>Run <code className="bg-white px-1 rounded text-xs">bash scripts/verify_study.sh study_XXX</code> and open a PR</span>
                </li>
              </ol>
              <p className="mt-4 text-xs text-gray-500">
                Maintainers assign final study numbering by merge order. Include your GitHub ID in <code className="bg-white px-0.5 rounded">index.json</code> contributors.
              </p>
            </div>

            {/* Web upload path */}
            <div className="rounded-2xl border border-gray-200 bg-gray-50/30 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Web Upload</h3>
              <p className="text-sm text-gray-600 mb-4">
                Upload a .zip of your study and we create a GitHub PR for you.
              </p>
              <ContributeUploadForm />
            </div>
          </div>

          {/* Docs link */}
          <div className="mt-8 flex items-center justify-center">
            <Link
              href="/docs/what_to_submit"
              className="group inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm text-gray-600 hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-700 transition-all"
            >
              <svg className="w-4 h-4 text-gray-400 group-hover:text-cyan-500 transition-colors" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
              Not sure where to start? Read the detailed walkthrough
              <svg className="w-3.5 h-3.5 text-gray-400 group-hover:text-cyan-500 transition-colors" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer note */}
      <div className="border-t border-gray-100 py-8">
        <p className="text-center text-sm text-gray-500">
          Questions? Open an issue on{" "}
          <a
            href="https://github.com/HumanStudy-Hub/HumanStudy-Bench"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-600 hover:underline"
          >
            GitHub
          </a>
          . For other inquiries, contact{" "}
          <a href="mailto:xul049@ucsd.edu" className="text-cyan-600 hover:underline">
            xul049@ucsd.edu
          </a>
          .
        </p>
      </div>
    </div>
  );
}
