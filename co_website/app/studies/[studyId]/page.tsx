import { promises as fs } from "fs";
import path from "path";
import Link from "next/link";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  const filePath = path.join(process.cwd(), "data/studies_index.json");
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const data = JSON.parse(raw);
    const studies = data.studies ?? [];
    return studies.map((s: { study_id: string }) => ({ studyId: s.study_id }));
  } catch {
    return [];
  }
}

type Contributor = { name: string; github?: string; institution?: string };
type StudyEntry = {
  study_id: string;
  title: string;
  authors: string[];
  year: number | null;
  description: string;
  contributors?: Contributor[];
};

export default async function StudyDetailPage({
  params,
}: {
  params: Promise<{ studyId: string }>;
}) {
  const { studyId } = await params;
  const filePath = path.join(process.cwd(), "data/studies_index.json");
  let studies: StudyEntry[] = [];
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const data = JSON.parse(raw);
    studies = data.studies ?? [];
  } catch (e) {
    console.error("Could not read studies index", e);
  }

  const study = studies.find((s) => s.study_id === studyId);
  if (!study) notFound();

  const authorsStr = study.authors?.length ? study.authors.join(", ") : "";
  const yearStr = study.year != null ? study.year : "";

  return (
    <div className="bg-white min-h-screen">
      <div className="py-16 sm:py-24 border-t border-gray-100">
        <div className="mx-auto max-w-4xl px-6 lg:px-8">
          <Link
            href="/studies"
            className="text-sm font-medium text-cyan-600 hover:text-cyan-500 mb-8 inline-block"
          >
            ← Back to studies
          </Link>

          <header className="mb-12">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">
              {study.title}
            </h1>
            <p className="mt-2 text-lg text-gray-600">
              {authorsStr}
              {yearStr ? ` (${yearStr})` : ""}
            </p>
            <p className="mt-2 text-sm text-gray-500 font-mono mb-6">{study.study_id}</p>
            
            <div className="flex flex-wrap gap-4">
              <a
                href={`https://github.com/HumanStudy-Hub/HumanStudy-Bench/tree/main/studies/${study.study_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-md bg-cyan-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-cyan-500 transition-colors"
              >
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" /></svg>
                View on GitHub
              </a>
            </div>
          </header>

          {study.description && (
            <section className="mb-12">
              <h2 className="text-xl font-semibold text-gray-900 font-serif mb-4">Description</h2>
              <p className="text-gray-600 leading-8 whitespace-pre-wrap">{study.description}</p>
            </section>
          )}

          {study.contributors && study.contributors.length > 0 && (
            <section className="mb-12">
              <h2 className="text-xl font-semibold text-gray-900 font-serif mb-4">Contributors</h2>
              <ul className="flex flex-wrap gap-3">
                {study.contributors.map((c) => (
                  <li key={c.name} className="flex items-baseline gap-1">
                    {c.github ? (
                      <a
                        href={c.github}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-600 hover:underline"
                      >
                        {c.name}
                      </a>
                    ) : (
                      <span className="text-gray-600">{c.name}</span>
                    )}
                    {c.institution && (
                      <span className="text-sm text-gray-400">({c.institution})</span>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="mb-12 rounded-xl border border-gray-200 bg-gray-50/50 p-6">
            <h2 className="text-xl font-semibold text-gray-900 font-serif mb-4">
              Repository Path
            </h2>
            <p className="text-sm font-mono text-gray-700">
              <span className="text-gray-500">Path:</span>{" "}
              <code className="bg-white px-1 rounded">studies/{study.study_id}/</code>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
