import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";

export default function BenchmarkPage() {
  const filePath = path.join(process.cwd(), "content/benchmark.md");
  const content = fs.readFileSync(filePath, "utf-8");

  return (
    <main className="prose mx-auto px-6 py-12">
      <ReactMarkdown>{content}</ReactMarkdown>
    </main>
  );
}
