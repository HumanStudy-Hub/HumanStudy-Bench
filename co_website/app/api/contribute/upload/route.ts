import { NextRequest, NextResponse } from "next/server";

const MAX_ZIP_BYTES = 50 * 1024 * 1024; // 50MB

function getStudyIdFromPath(entryName: string): string | null {
  const m = entryName.match(/^(?:data\/studies\/)?(study_\d{3})\//) || entryName.match(/^(study_\d{3})\//);
  return m ? m[1] : null;
}

function parseJson(buf: Buffer): unknown {
  return JSON.parse(buf.toString("utf8"));
}

function normalizeGithub(input: string): string {
  const raw = input.trim();
  if (!raw) return "";
  if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
  return `https://github.com/${raw.replace(/^@/, "")}`;
}

type ValidationResult = { ok: true; studyId: string; files: Map<string, Buffer> } | { ok: false; errors: string[] };

function validateZip(entries: { name: string; data: Buffer }[]): ValidationResult {
  const errors: string[] = [];
  const fileMap = new Map<string, Buffer>();
  for (const e of entries) {
    if (e.data.length) fileMap.set(e.name, e.data);
  }

  let studyId: string | null = null;
  for (const name of fileMap.keys()) {
    const id = getStudyIdFromPath(name) || (name.startsWith("study_") && name.match(/^study_\d{3}/)?.[0]);
    if (id) {
      studyId = id;
      break;
    }
  }
  if (!studyId) {
    const dataStudy = [...fileMap.keys()].find((k) => k.includes("studies/study_"));
    const match = dataStudy?.match(/study_\d{3}/);
    studyId = match ? match[0] : null;
  }
  if (!studyId) {
    errors.push("Zip must contain a study_XXX directory (e.g. studies/study_014/ or study_014/).");
    return { ok: false, errors };
  }

  const dataPrefix = `studies/${studyId}/`;
  const dataPrefixAlt = `${studyId}/`;
  const getData = (path: string) =>
    fileMap.get(dataPrefix + path) || fileMap.get(dataPrefixAlt + path);

  const specBuf = getData("specification.json");
  const metaBuf = getData("metadata.json");
  const gtBuf = getData("ground_truth.json");
  if (!specBuf) errors.push("Missing specification.json");
  if (!metaBuf) errors.push("Missing metadata.json");
  if (!gtBuf) errors.push("Missing ground_truth.json");

  let spec: Record<string, unknown>;
  let meta: Record<string, unknown>;
  let gt: Record<string, unknown>;
  try {
    spec = specBuf ? (parseJson(specBuf) as Record<string, unknown>) : {};
    meta = metaBuf ? (parseJson(metaBuf) as Record<string, unknown>) : {};
    gt = gtBuf ? (parseJson(gtBuf) as Record<string, unknown>) : {};
  } catch {
    errors.push("Invalid JSON in specification, metadata, or ground_truth.");
    return { ok: false, errors };
  }

  if (spec) {
    if (!spec.participants || typeof (spec.participants as Record<string, unknown>)?.n !== "number")
      errors.push("specification.json must have participants.n (number).");
    if (!spec.design) errors.push("specification.json must have design.");
    if (!spec.procedure) errors.push("specification.json must have procedure.");
  }
  if (meta) {
    if (!meta.id && !meta.title) errors.push("metadata.json must have id or title.");
    if (!meta.authors) errors.push("metadata.json must have authors.");
    if (!meta.domain) errors.push("metadata.json must have domain.");
    if (!meta.findings) errors.push("metadata.json must have findings.");
  }
  const studies = gt?.studies as Array<Record<string, unknown>> | undefined;
  if (!Array.isArray(studies) || studies.length === 0) {
    errors.push("ground_truth.json must have a non-empty studies array.");
  }

  const materialsEntries = [...fileMap.entries()].filter(
    ([n]) => n.startsWith(dataPrefix + "materials/") || n.startsWith(dataPrefixAlt + "materials/")
  );
  if (materialsEntries.length === 0) errors.push("At least one materials/*.json file is required.");
  const gtKeys = new Set<string>();
  if (Array.isArray(studies)) {
    for (const s of studies) {
      const findings = s.findings as Array<Record<string, unknown>> | undefined;
      if (!Array.isArray(findings)) continue;
      for (const f of findings) {
        const odp = f.original_data_points as Record<string, unknown> | undefined;
        const data = odp?.data as Record<string, unknown> | undefined;
        if (data && typeof data === "object") Object.keys(data).forEach((k) => gtKeys.add(k));
      }
    }
  }
  for (const [name, buf] of materialsEntries) {
    if (!name.endsWith(".json")) continue;
    try {
      const mat = parseJson(buf) as Record<string, unknown>;
      if (!mat.sub_study_id) errors.push(`Materials file ${name} must have sub_study_id.`);
      if (!mat.instructions) errors.push(`Materials file ${name} must have instructions.`);
      const items = mat.items as Array<Record<string, unknown>> | undefined;
      if (!Array.isArray(items)) {
        errors.push(`Materials file ${name} must have items array.`);
      } else {
        for (const item of items) {
          const metaItem = item.metadata as Record<string, unknown> | undefined;
          const gk = metaItem?.gt_key as string | undefined;
          if (gk && !gtKeys.has(gk)) errors.push(`gt_key "${gk}" in ${name} not found in ground_truth data.`);
        }
      }
    } catch {
      errors.push(`Invalid JSON in materials file ${name}.`);
    }
  }

  const configName = `src/studies/${studyId}_config.py`;
  const evaluatorName = `src/studies/${studyId}_evaluator.py`;
  const hasConfig = fileMap.has(configName) || [...fileMap.keys()].some((k) => k.endsWith(`${studyId}_config.py`));
  const hasEvaluator = fileMap.has(evaluatorName) || [...fileMap.keys()].some((k) => k.endsWith(`${studyId}_evaluator.py`));
  if (!hasConfig) errors.push(`Missing ${studyId}_config.py in src/studies/ or zip root.`);
  if (!hasEvaluator) errors.push(`Missing ${studyId}_evaluator.py in src/studies/ or zip root.`);

  if (errors.length > 0) return { ok: false, errors };
  return { ok: true, studyId, files: fileMap };
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const contributorGithub = normalizeGithub(String(formData.get("contributor_github") || ""));
    const contributorName = String(formData.get("contributor_name") || "").trim();
    const contributorInstitution = String(formData.get("contributor_institution") || "Independent Researcher").trim();
    if (!file || !(file instanceof File)) {
      return NextResponse.json({ success: false, errors: ["No file provided."] }, { status: 400 });
    }
    if (!contributorGithub) {
      return NextResponse.json(
        { success: false, errors: ["GitHub ID or profile URL is required."] },
        { status: 400 }
      );
    }
    if (!file.name.endsWith(".zip")) {
      return NextResponse.json({ success: false, errors: ["File must be a .zip archive."] }, { status: 400 });
    }
    const buf = Buffer.from(await file.arrayBuffer());
    if (buf.length > MAX_ZIP_BYTES) {
      return NextResponse.json({ success: false, errors: ["Zip file exceeds 50MB limit."] }, { status: 400 });
    }

    const AdmZip = (await import("adm-zip")).default;
    const zip = new AdmZip(buf);
    const entries = zip.getEntries();
    const extracted: { name: string; data: Buffer }[] = [];
    for (const e of entries) {
      if (e.isDirectory) continue;
      extracted.push({ name: e.entryName.replace(/\/$/, ""), data: e.getData() });
    }
    const result = validateZip(extracted);
    if (!result.ok) {
      return NextResponse.json({ success: false, errors: result.errors }, { status: 400 });
    }
    const { studyId, files: validatedFiles } = result;

    const token = process.env.GITHUB_TOKEN;
    const repo = process.env.GITHUB_REPO || "HumanStudy-Hub/HumanStudy-Bench";
    if (!token) {
      return NextResponse.json(
        { success: false, errors: ["Server is not configured with GITHUB_TOKEN for creating PRs."] },
        { status: 503 }
      );
    }

    const [owner, repoName] = repo.split("/");
    const { Octokit } = await import("@octokit/rest");
    const octokit = new Octokit({ auth: token });

    const { data: defaultBranch } = await octokit.repos.get({ owner, repo: repoName });
    const baseRef = defaultBranch.default_branch || "main";
    const branchName = `contribute/${studyId}-${Date.now()}`;

    const { data: baseRefData } = await octokit.git.getRef({ owner, repo: repoName, ref: `heads/${baseRef}` });
    const baseCommitSha = baseRefData.object.sha;
    const { data: commitData } = await octokit.git.getCommit({ owner, repo: repoName, commit_sha: baseCommitSha });
    const baseTreeSha = commitData.tree.sha;

    function zipEntryToRepoPath(entryName: string): string | null {
      const sid = studyId;
      if (entryName.startsWith(`studies/${sid}/`)) return entryName;
      if (entryName.startsWith(`${sid}/`)) return `studies/${entryName}`;
      if (entryName === `src/studies/${sid}_config.py` || entryName.endsWith(`/${sid}_config.py`)) return `src/studies/${sid}_config.py`;
      if (entryName === `src/studies/${sid}_evaluator.py` || entryName.endsWith(`/${sid}_evaluator.py`)) return `src/studies/${sid}_evaluator.py`;
      if (entryName.includes("specification.json")) return `studies/${sid}/specification.json`;
      if (entryName.includes("metadata.json")) return `studies/${sid}/metadata.json`;
      if (entryName.includes("ground_truth.json")) return `studies/${sid}/ground_truth.json`;
      if (entryName.includes("materials/")) return `studies/${sid}/materials/${entryName.split("materials/")[1]}`;
      return null;
    }

    const pathToContent = new Map<string, Buffer>();
    for (const [name, data] of validatedFiles.entries()) {
      if (!data.length) continue;
      const repoPath = zipEntryToRepoPath(name);
      if (repoPath && (repoPath.startsWith("studies/") || repoPath.startsWith("src/"))) {
        pathToContent.set(repoPath, data);
      }
    }

    const metaBuf = pathToContent.get(`studies/${studyId}/metadata.json`);
    const meta = metaBuf ? (parseJson(metaBuf) as Record<string, unknown>) : {};
    const title = (meta.title as string) || studyId;

    const generatedIndex = {
      title: title,
      authors: Array.isArray(meta.authors) ? meta.authors : [],
      year: typeof meta.year === "number" ? meta.year : null,
      description: typeof meta.summary === "string" ? meta.summary : typeof meta.abstract === "string" ? meta.abstract : "",
      contributors: [{ name: contributorName || undefined, github: contributorGithub, institution: contributorInstitution }],
    };
    pathToContent.set(
      `studies/${studyId}/index.json`,
      Buffer.from(JSON.stringify(generatedIndex, null, 2), "utf8")
    );

    const treeItems: { path: string; mode: "100644"; type: "blob"; sha: string }[] = [];
    for (const [path, data] of pathToContent.entries()) {
      const { data: blob } = await octokit.git.createBlob({
        owner,
        repo: repoName,
        content: data.toString("base64"),
        encoding: "base64",
      });
      treeItems.push({ path, mode: "100644", type: "blob", sha: blob.sha });
    }

    const { data: newTree } = await octokit.git.createTree({
      owner,
      repo: repoName,
      base_tree: baseTreeSha,
      tree: treeItems,
    });

    const { data: newCommit } = await octokit.git.createCommit({
      owner,
      repo: repoName,
      message: `chore: add ${studyId} (contribute upload)`,
      tree: newTree.sha,
      parents: [baseCommitSha],
    });

    await octokit.git.createRef({
      owner,
      repo: repoName,
      ref: `refs/heads/${branchName}`,
      sha: newCommit.sha,
    });

    const prBody = `## Contributed study (website upload)\n\n- **Study ID:** ${studyId}\n- **Title:** ${title}\n- **GitHub:** ${contributorGithub}\n- **Domain:** ${meta.domain || "—"}\n\nPlease run \`python scripts/validate_study.py --study ${studyId}\` and review.`;

    const pr = await octokit.pulls.create({
      owner,
      repo: repoName,
      title: `Add ${studyId}: ${title}`,
      head: branchName,
      base: baseRef,
      body: prBody,
    });

    return NextResponse.json({ success: true, pr_url: pr.data.html_url });
  } catch (err) {
    console.error("Contribute upload error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { success: false, errors: [message] },
      { status: 500 }
    );
  }
}
