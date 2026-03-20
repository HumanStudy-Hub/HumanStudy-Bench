import { promises as fs } from "fs";
import path from "path";
import Link from "next/link";

type Contributor = { name: string; github?: string; institution?: string };
type StudySummary = {
  study_id: string;
  title: string;
  authors: string[];
  year: number | null;
  contributors?: Contributor[];
};

export default async function StudiesPage() {
  const filePath = path.join(process.cwd(), "data/studies_index.json");
  let studies: StudySummary[] = [];
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const data = JSON.parse(raw);
    studies = data.studies ?? [];
  } catch (e) {
    console.error("Could not read studies index", e);
  }

  return (
    <div className="bg-white min-h-screen">
      <div className="py-12 border-t border-gray-100">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center mb-8">
            <h1 className="text-xl font-bold tracking-tight text-gray-900 sm:text-2xl font-serif">
              Study Catalog
            </h1>
            <p className="mt-1 text-gray-500 text-xs">
              Standardized human study replays.
            </p>
          </div>
          
          <div className="mx-auto grid max-w-2xl grid-cols-1 gap-3 lg:max-w-none lg:grid-cols-2">
            {studies.slice(0, 12).map((study, idx) => (
              <div key={study.study_id} className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-3 py-2.5 hover:border-cyan-300 hover:shadow-sm transition-all">
                <div className="flex items-center gap-3 overflow-hidden">
                  <span className="inline-flex h-6 min-w-8 items-center justify-center rounded-md border border-cyan-100 bg-cyan-50 px-1 text-[10px] font-bold text-cyan-700 shrink-0">
                    {(idx + 1).toString().padStart(3, '0')}
                  </span>
                  <Link
                    href={`/studies/${study.study_id}`}
                    className="text-xs font-semibold text-gray-900 hover:text-cyan-700 transition-colors truncate"
                  >
                    {study.title}
                  </Link>
                </div>

                {study.contributors?.[0] && (
                  <div className="flex items-center gap-2 shrink-0 ml-4">
                    {study.contributors[0].github ? (
                      <a
                        href={study.contributors[0].github}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-gray-500 hover:text-cyan-700 transition-colors"
                      >
                        @{study.contributors[0].github}
                      </a>
                    ) : (
                      <span className="text-[10px] text-gray-500">
                        {study.contributors[0].name}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {studies.length > 12 && (
            <>
              <div className="mx-auto max-w-2xl lg:max-w-none my-8 flex items-center gap-4">
                <div className="flex-1 border-t border-gray-300" />
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Community Contributions</span>
                <div className="flex-1 border-t border-gray-300" />
              </div>

              <div className="mx-auto grid max-w-2xl grid-cols-1 gap-3 lg:max-w-none lg:grid-cols-2">
                {studies.slice(12).map((study, idx) => (
                  <div key={study.study_id} className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-3 py-2.5 hover:border-cyan-300 hover:shadow-sm transition-all">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <span className="inline-flex h-6 min-w-8 items-center justify-center rounded-md border border-cyan-100 bg-cyan-50 px-1 text-[10px] font-bold text-cyan-700 shrink-0">
                        {(idx + 13).toString().padStart(3, '0')}
                      </span>
                      <Link
                        href={`/studies/${study.study_id}`}
                        className="text-xs font-semibold text-gray-900 hover:text-cyan-700 transition-colors truncate"
                      >
                        {study.title}
                      </Link>
                    </div>

                    {study.contributors?.[0] && (
                      <div className="flex items-center gap-2 shrink-0 ml-4">
                        {study.contributors[0].github ? (
                          <a
                            href={study.contributors[0].github}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] text-gray-500 hover:text-cyan-700 transition-colors"
                          >
                            @{study.contributors[0].github}
                          </a>
                        ) : (
                          <span className="text-[10px] text-gray-500">
                            {study.contributors[0].name}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
