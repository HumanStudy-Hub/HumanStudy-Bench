"use client";

import { useState, useCallback } from "react";

type Result = { success: true; pr_url: string } | { success: false; errors: string[] };

export default function ContributeUploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [contributorGithub, setContributorGithub] = useState("");
  const [contributorName, setContributorName] = useState("");
  const [contributorInstitution, setContributorInstitution] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.name.endsWith(".zip")) setFile(f);
    else setResult({ success: false, errors: ["Please upload a .zip file."] });
  }, []);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f?.name.endsWith(".zip")) {
      setFile(f);
      setResult(null);
    } else if (f) {
      setResult({ success: false, errors: ["Please select a .zip file."] });
    }
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!file) {
        setResult({ success: false, errors: ["Please select a zip file first."] });
        return;
      }
      setLoading(true);
      setResult(null);
      const formData = new FormData();
      formData.append("file", file);
      formData.append("contributor_github", contributorGithub.trim());
      formData.append("contributor_name", contributorName.trim());
      formData.append("contributor_institution", contributorInstitution.trim() || "Independent Researcher");
      try {
        const res = await fetch("/api/contribute/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.success) setResult({ success: true, pr_url: data.pr_url });
        else setResult({ success: false, errors: data.errors || [data.error] || ["Upload failed."] });
      } catch {
        setResult({ success: false, errors: ["Network error. Please try again."] });
      } finally {
        setLoading(false);
      }
    },
    [file, contributorGithub, contributorName, contributorInstitution]
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <input
        type="text"
        value={contributorName}
        onChange={(e) => setContributorName(e.target.value)}
        placeholder="Your name (required)"
        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none"
        required
      />
      <input
        type="text"
        value={contributorGithub}
        onChange={(e) => setContributorGithub(e.target.value)}
        placeholder="GitHub ID or profile URL (required)"
        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none"
        required
      />
      <input
        type="text"
        value={contributorInstitution}
        onChange={(e) => setContributorInstitution(e.target.value)}
        placeholder="Institution (leave blank for Independent Researcher)"
        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none"
      />
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragActive ? "border-cyan-500 bg-cyan-50/50" : "border-gray-300 bg-gray-50/50"
        }`}
      >
        <input
          type="file"
          accept=".zip"
          onChange={handleChange}
          className="hidden"
          id="contribute-zip"
        />
        <label htmlFor="contribute-zip" className="cursor-pointer">
          <p className="text-sm font-medium text-gray-700">
            {file ? file.name : "Drag and drop your study .zip here, or click to choose"}
          </p>
          <p className="text-xs text-gray-500 mt-1">Max 50MB. Zip should contain index.json, source/, scripts/, and README.md.</p>
        </label>
      </div>
      <button
        type="submit"
        disabled={!file || loading}
        className="w-full rounded-md bg-cyan-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Validating and creating PR…" : "Upload and create PR"}
      </button>
      {result && (
        <div
          className={`rounded-lg p-4 text-sm ${
            result.success ? "bg-green-50 text-green-800 border border-green-200" : "bg-red-50 text-red-800 border border-red-200"
          }`}
        >
          {result.success ? (
            <p>
              PR created:{" "}
              <a href={result.pr_url} target="_blank" rel="noopener noreferrer" className="font-medium underline">
                {result.pr_url}
              </a>
            </p>
          ) : (
            <ul className="list-disc list-inside space-y-1">
              {result.errors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </form>
  );
}
