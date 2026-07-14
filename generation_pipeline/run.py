"""
CLI for ai-ethics Generation Pipeline (Stages 1 & 2).

Usage:
    # Stage 1 — filter a single PDF in current dir or data/papers/
    python generation_pipeline/run.py --stage 1 --pdf "data/papers/2024_Xu_Petty.pdf"

    # Stage 1 source discovery/fetch — find OSF links from Stage 1 JSON/PDF and fetch files
    python generation_pipeline/run.py --stage 1 --pdf "data/papers/2024_Xu_Petty.pdf" --fetch-sources

    # Stage 2 — extract effects (uses latest stage1 JSON for the paper)
    python generation_pipeline/run.py --stage 2 --pdf "data/papers/2024_Xu_Petty.pdf"

    # Refine: re-run stage based on a manually-edited review .md
    python generation_pipeline/run.py --stage 2 --pdf <pdf> --refine

    # Stage 4 - build HumanStudy-Bench study package
    python generation_pipeline/run.py --stage 4 --json outputs/<paper>/stage3.json --study-id study_my_paper

    # Stage 5 - simulate participants with the generated HumanStudy-Bench adapter
    python generation_pipeline/run.py --stage 5 --study-id study_my_paper --mock-agent

    # Verification gate for already extracted JSON
    python generation_pipeline/run.py --verify --json stage4/runnable/<paper>/*.json --pdf stage4/runnable/<paper>/*.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_pdf(pdf_arg: str | list[str] | None) -> Path:
    if isinstance(pdf_arg, list):
        if len(pdf_arg) == 0:
            pdf_arg = None
        elif len(pdf_arg) == 1:
            pdf_arg = pdf_arg[0]
        else:
            pdf_arg = " ".join(pdf_arg)
    if pdf_arg:
        p = Path(pdf_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"PDF not found: {p}")
    # Fallback: look in data/papers/ then cwd
    for d in (Path("data/papers"), Path.cwd()):
        if d.exists():
            pdfs = list(d.glob("*.pdf"))
            if len(pdfs) == 1:
                return pdfs[0]
            if len(pdfs) > 1:
                raise ValueError(f"Multiple PDFs in {d}, use --pdf to pick one.")
    raise FileNotFoundError("No PDF found. Pass --pdf <path>.")


def find_latest_stage_file(stage: int, output_dir: Path, paper_id: str | None = None) -> Path:
    filename = f"stage{stage}.json"
    candidates: list[Path] = [output_dir / filename]
    if paper_id:
        candidates.insert(0, output_dir / paper_id / filename)
    else:
        candidates.extend(output_dir.glob(f"*/{filename}"))
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No canonical {filename} in {output_dir} (paper_id={paper_id})"
        )
    return max(files, key=lambda p: p.stat().st_mtime)


def print_source_fetch_results(results: list[dict]) -> None:
    ok = sum(1 for item in results if item["summary"]["errors"] == 0)
    for item in results:
        summary = item["summary"]
        reused = summary.get("reused_files", 0)
        reused_part = f" reused={reused}" if reused else ""
        print(
            f"{Path(item['json_path']).parent.name}: "
            f"plans={summary['plans']} fetched={summary['fetched_files']} "
            f"errors={summary['errors']}{reused_part} sources={item['sources_dir']}"
        )
    print(f"Source fetch complete: {ok}/{len(results)} paper(s) without connector errors.")


def _stage4_llm_requested(settings) -> bool:
    for section_name in ("stage4_llm", "config_llm", "gym_llm"):
        section = settings.section(section_name)
        for key in ("provider", "model", "api_key", "base_url", "api_base"):
            value = section.get(key)
            if value not in (None, ""):
                return True
    return False


def _first_cli_or_setting(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "default", "default(unset)"}:
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="ai-ethics extraction pipeline (stages 1, 2, 3, 4, 5)")
    parser.add_argument("--settings", type=Path, help="Path to settings YAML/JSON (default: config/settings.yaml if present)")
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "5"],
                        help="1=Filter, 2=Extraction, 3=Patch, 4=Human study package, 5=Human simulation")
    parser.add_argument("--pdf", nargs="*", help="Path to PDF file(s)")
    parser.add_argument("--json", nargs="+", help="Per-paper JSON file(s), used by --verify")
    parser.add_argument(
        "--fetch-sources",
        action="store_true",
        help=(
            "Fetch OSF sources. With --stage, this is Stage 1 only; without --stage, "
            "fetch explicit --json/--pdf pairs or stage4 corpus files."
        ),
    )
    parser.add_argument("--stage4-dir", type=Path, default=Path("stage4"), help="Root stage4 corpus directory")
    parser.add_argument("--osf-token", default=None, help="OSF API bearer token; defaults to OSF_TOKEN env var")
    parser.add_argument("--verify", action="store_true", help="Run schema + hallucination verification gate")
    parser.add_argument("--verification-report", type=Path, help="Optional aggregate verification report path")
    parser.add_argument("--source", nargs="*", default=[], help="Additional source text files for --verify")
    parser.add_argument("--source-dir", nargs="*", default=[], help="Additional source directories for --verify")
    parser.add_argument("--refine", action="store_true", help="Re-run stage after manual review")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiments-dir", type=Path, help="Deprecated alias for Stage 4 data directory")
    parser.add_argument("--data-dir", type=Path, help="HumanStudy-Bench data directory")
    parser.add_argument(
        "--hub-layout",
        action="store_true",
        help="Stage 4: write a HumanStudy-Bench-Hub study folder with source/ and scripts/",
    )
    parser.add_argument(
        "--hub-studies-dir",
        type=Path,
        default=Path("studies"),
        help="Stage 4 --hub-layout output directory for studies/<study_id>",
    )
    parser.add_argument("--runs-dir", type=Path, help="Stage 5 run output directory")
    parser.add_argument("--study-id", help="HumanStudy-Bench study id for Stage 4/5")
    parser.add_argument("--provider", choices=["openai", "google", "anthropic", "openrouter", "vllm"])
    parser.add_argument("--model")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", "--base-url", dest="api_base", default=None)
    parser.add_argument("--regeneration-instructions", type=str, help="Path to regeneration feedback JSON (stage 2)")
    parser.add_argument("--no-stage1-verifier", action="store_true", help="Stage 1: skip LLM study-inventory verifier pass")
    parser.add_argument(
        "--stage1-verifier-timeout",
        type=float,
        default=60.0,
        help="Stage 1 verifier request timeout in seconds; <=0 disables timeout",
    )
    parser.add_argument(
        "--stage1-auto-refine-attempts",
        type=int,
        default=1,
        help="Stage 1: automatically rerun inventory extraction with verifier feedback this many times",
    )
    parser.add_argument(
        "--no-stage1-auto-refine",
        action="store_true",
        help="Stage 1: disable automatic verifier-feedback reruns",
    )
    parser.add_argument(
        "--stage2-grounding",
        action="store_true",
        help="Legacy Stage 2: run slow per-effect material-slot grounding. Disabled by default; Stage 3 owns material evidence.",
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Deprecated compatibility flag; Stage 2 grounding is already disabled unless --stage2-grounding is set.",
    )
    parser.add_argument("--no-stage2-verifier", action="store_true", help="Stage 2: skip LLM study/finding verifier pass")
    parser.add_argument(
        "--stage2-verifier-timeout",
        type=float,
        default=60.0,
        help="Stage 2 verifier request timeout in seconds; <=0 disables timeout",
    )
    parser.add_argument(
        "--stage2-auto-refine-attempts",
        type=int,
        default=1,
        help="Stage 2: automatically rerun extraction with verifier feedback this many times",
    )
    parser.add_argument(
        "--no-stage2-auto-refine",
        action="store_true",
        help="Stage 2: disable automatic verifier-feedback reruns",
    )
    parser.add_argument("--ground-threshold", type=float, default=90.0, help="Legacy Stage 2 grounding verbatim threshold")
    parser.add_argument("--ground-k", type=int, default=8, help="Legacy Stage 2 grounding retrieval chunks per slot")
    parser.add_argument(
        "--ground-timeout",
        type=float,
        default=60.0,
        help="Legacy Stage 2 helper timeout in seconds for each grounding LLM call; <=0 disables timeout",
    )
    parser.add_argument(
        "--ground-workers",
        type=int,
        default=4,
        help="Legacy Stage 2 parallel grounding workers; use 1 for serial execution",
    )
    # Stage 3 options
    parser.add_argument("--json-dir", type=str, help="Directory of per-paper JSONs to patch (stage 3)")
    parser.add_argument("--pdf-dir", type=str, help="Directory of PDFs to pair with JSONs (stage 3)")
    parser.add_argument(
        "--osf-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "OSF/source directory for Stage 3. May point to a sources/ directory "
            "containing combined_sources.txt or a paper directory containing sources/. "
            "Repeatable. If omitted, Stage 3 auto-checks JSON and PDF sibling sources/."
        ),
    )
    parser.add_argument("--only", action="append", default=None,
                        help="Filename substring filter — only patch matching JSONs (repeatable, stage 3)")
    parser.add_argument("--overwrite-filled", action="store_true",
                        help="Re-run already-filled slots (default: preserve existing fills)")
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak backups when overwriting")
    parser.add_argument("--no-extract-text", action="store_true", help="Skip source text extraction after fetching")
    parser.add_argument("--dry-run", action="store_true", help="Pair + call LLM but don't write back")
    parser.add_argument(
        "--stage3-slot-fill",
        action="store_true",
        help="Legacy Stage 3: run PDF slot filling before source/material assembly",
    )
    parser.add_argument(
        "--allow-effect-slot-fallback",
        action="store_true",
        help="Legacy Stage 3: allow per-effect materials/manipulation/items slots as material fallback",
    )
    parser.add_argument(
        "--stage3-select-votes",
        type=int,
        default=3,
        help="Stage 3 LLM study-selection votes; lower to 1 for small smoke tests",
    )
    parser.add_argument(
        "--stage3-select-timeout",
        type=float,
        default=60.0,
        help="Stage 3 per-call LLM study-selection timeout in seconds; <=0 disables timeout",
    )
    parser.add_argument(
        "--stage3-pdf-timeout",
        type=float,
        default=120.0,
        help="Stage 3 PDF material extraction timeout in seconds; <=0 disables timeout",
    )
    parser.add_argument("--max-repair-iters", type=int, default=3, help="Deprecated old gym Stage 4 option; ignored")
    parser.add_argument("--no-generate-config", action="store_true", help="Stage 4: skip LLM-generated StudyConfig adapter")
    # Stage 5 options
    parser.add_argument("--experiment", type=Path, help="Deprecated Stage 5 alias for --study-id or study directory")
    parser.add_argument("--sim-models", nargs="+", help="Stage 5 agent model(s); defaults to stage5_llm.model")
    parser.add_argument("--n-agents", type=int, help="Override Stage 5 agent count")
    parser.add_argument("--max-agents", type=int, help="Explicit Stage 5 cost cap; omitted means no cap")
    parser.add_argument("--repeats", type=int, help="Stage 5 repeats per persona")
    parser.add_argument("--temperature", type=float, help="Stage 5 agent LLM temperature")
    parser.add_argument("--seed", type=int, help="Stage 5 deterministic seed")
    parser.add_argument("--careless-rate", type=float, help="Deprecated old gym Stage 5 option; ignored")
    parser.add_argument("--parse-failure-threshold", type=float, help="Deprecated old gym Stage 5 option; ignored")
    parser.add_argument("--min-condition-n", type=int, help="Deprecated old gym Stage 5 option; ignored")
    parser.add_argument("--include-partial", action="store_true", help="Deprecated old gym Stage 5 option; ignored")
    parser.add_argument("--mock-agent", action="store_true", help="Stage 5 use deterministic mock policy instead of LLM")
    parser.add_argument("--num-workers", type=int, help="Stage 5 parallel participant workers")
    parser.add_argument("--use-cache", action="store_true", help="Stage 5 cache raw repeat results")
    parser.add_argument("--cache-dir", type=Path, help="Stage 5 cache directory")
    parser.add_argument("--profiles-json", type=Path, help="Stage 5 participant profiles JSON")
    parser.add_argument("--system-prompt-file", type=Path, help="Stage 5 custom system prompt file")
    parser.add_argument("--system-prompt-preset", help="Stage 5 system prompt preset")
    parser.add_argument("--reasoning", help="Stage 5 reasoning setting for compatible models")
    parser.add_argument("--enable-reasoning", action="store_true", help="Stage 5 force reasoning for compatible models")
    parser.add_argument(
        "--allow-unready",
        action="store_true",
        help="Stage 5: explicitly allow a generated package whose audit requires human patching",
    )
    args = parser.parse_args()

    from generation_pipeline.settings import (
        load_settings,
        resolve_data_dir,
        resolve_output_dir,
        resolve_runs_dir,
        resolve_stage_llm_config,
        resolve_verification_threshold,
    )

    settings = load_settings(args.settings)
    output_dir = resolve_output_dir(settings, args.output_dir)
    verification_threshold = resolve_verification_threshold(settings)

    if args.fetch_sources and not args.stage:
        from generation_pipeline.connectors.registry import fetch_json_sources, fetch_stage4_sources
        from src.llm.factory import get_client as _get_client

        # Build an LLM client for link-discovery fallback (papers with no explicit OSF URL).
        try:
            _llm_cfg = resolve_stage_llm_config(
                settings,
                stage=1,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                api_base=args.api_base,
            )
            _llm_client = _get_client(provider=_llm_cfg.provider, model=_llm_cfg.model,
                                      api_key=_llm_cfg.api_key, api_base=_llm_cfg.api_base)
        except Exception:
            _llm_client = None  # no key configured — skip LLM fallback silently

        if args.json:
            results = fetch_json_sources(
                [Path(item) for item in args.json],
                pdf_paths=[Path(item) for item in (args.pdf or [])],
                token=args.osf_token,
                dry_run=args.dry_run,
                extract_text=not args.no_extract_text,
                output_dir=args.output_dir,
                llm_client=_llm_client,
            )
        else:
            results = fetch_stage4_sources(
                args.stage4_dir,
                only=args.only,
                token=args.osf_token,
                dry_run=args.dry_run,
                extract_text=not args.no_extract_text,
                llm_client=_llm_client,
            )
        print_source_fetch_results(results)
        if any(item["summary"]["errors"] for item in results):
            sys.exit(1)
        return

    if args.verify:
        if not args.json:
            raise SystemExit("--verify requires --json <paper.json> [more.json]")
        from generation_pipeline.verification.verbatim_verifier import verify_files

        aggregate = verify_files(
            [Path(item) for item in args.json],
            pdf_paths=[Path(item) for item in (args.pdf or [])],
            source_paths=[Path(item) for item in args.source],
            source_dirs=[Path(item) for item in args.source_dir],
            threshold=verification_threshold,
            write=not args.dry_run,
            backup=not args.no_backup,
            aggregate_report_path=args.verification_report,
            repair=not args.dry_run,
        )
        print(f"Verification complete: {aggregate['summary']}")
        if aggregate["summary"]["failed"]:
            sys.exit(1)
        return

    if not args.stage:
        parser.error("--stage is required unless --verify is used")

    stage = int(args.stage)
    if args.fetch_sources and stage != 1:
        raise SystemExit(
            "--fetch-sources belongs to Stage 1 in the staged pipeline. "
            "Run `--stage 1 --fetch-sources` before Stage 2, or use standalone "
            "`--fetch-sources --json <paper.json> --pdf <paper.pdf>` for old JSONs."
        )

    if stage == 5:
        study_ref = args.study_id or args.experiment
        if not study_ref:
            raise SystemExit("Stage 5 requires --study-id <study_id> or --experiment <study_id|study_dir>")
        from generation_pipeline.stage5 import Stage5Options, run_stage5

        runs_dir = resolve_runs_dir(settings, args.runs_dir)
        stage5_cfg = resolve_stage_llm_config(
            settings,
            stage=5,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
        )
        models = args.sim_models or [stage5_cfg.model]

        stage5_settings = settings.section("stage5")
        options = Stage5Options(
            n_participants=_first_cli_or_setting(args.n_agents, stage5_settings.get("n_participants")),
            n_agents=_first_cli_or_setting(args.n_agents, stage5_settings.get("n_agents")),
            max_agents=_first_cli_or_setting(args.max_agents, stage5_settings.get("max_agents")),
            repeats=int(_first_cli_or_setting(args.repeats, stage5_settings.get("repeats"), 1)),
            temperature=_optional_float(_first_cli_or_setting(args.temperature, stage5_settings.get("temperature"))) or 1.0,
            seed=int(_first_cli_or_setting(args.seed, stage5_settings.get("seed"), 42)),
            dry_run=args.dry_run,
            mock=args.mock_agent,
            num_workers=_first_cli_or_setting(args.num_workers, stage5_settings.get("num_workers")),
            use_cache=bool(args.use_cache or stage5_settings.get("use_cache", False)),
            cache_dir=_first_cli_or_setting(args.cache_dir, stage5_settings.get("cache_dir")),
            profiles_json=_first_cli_or_setting(args.profiles_json, stage5_settings.get("profiles_json")),
            system_prompt_file=_first_cli_or_setting(args.system_prompt_file, stage5_settings.get("system_prompt_file")),
            system_prompt_preset=_first_cli_or_setting(
                args.system_prompt_preset,
                stage5_settings.get("system_prompt_preset"),
                "v3_human_plus_demo",
            ),
            reasoning=_first_cli_or_setting(args.reasoning, stage5_settings.get("reasoning"), "default"),
            enable_reasoning=bool(args.enable_reasoning or stage5_settings.get("enable_reasoning", False)),
            allow_unready=bool(args.allow_unready),
            api_key=stage5_cfg.api_key,
            api_base=stage5_cfg.api_base,
        )
        print(f"Running Stage 5: HumanStudy-Bench simulation for {study_ref}")
        summary = run_stage5(
            study_ref,
            runs_dir=runs_dir,
            models=models,
            options=options,
            data_dir=resolve_data_dir(settings, args.data_dir),
        )
        print(
            "✓ Stage 5 complete: "
            f"study_id={summary['study_id']} models={summary['completed']} "
            f"runs={summary['run_count']} use_real_llm={summary['use_real_llm']}"
        )
        print(f"  Runs: {runs_dir / summary['study_id']}")
        return

    if stage == 4:
        if not args.json:
            raise SystemExit("Stage 4 requires --json <patched stage2 JSON> [more.json]")
        if args.study_id and len(args.json) > 1:
            raise SystemExit("--study-id can only be used with one Stage 4 JSON input")
        from generation_pipeline.stage4 import build_human_study_package, paper_local_stage4_dir
        from src.llm.factory import get_client as _get_client

        explicit_data_dir = args.experiments_dir or args.data_dir
        if args.hub_layout:
            data_dir = args.hub_studies_dir
        else:
            data_dir = resolve_data_dir(settings, explicit_data_dir) if explicit_data_dir else None
        stage4_generation_llm_requested = _stage4_llm_requested(settings) or any(
            value is not None for value in (args.provider, args.model, args.api_key, args.api_base)
        )
        _llm_cfg = resolve_stage_llm_config(
            settings,
            stage=4,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
        )
        _selection_llm_client = None
        if _llm_cfg.api_key:
            _selection_llm_client = _get_client(
                provider=_llm_cfg.provider,
                model=_llm_cfg.model,
                api_key=_llm_cfg.api_key,
                api_base=_llm_cfg.api_base,
            )
        pdf_path = None
        if args.pdf and len(args.pdf) == 1:
            pdf_path = Path(args.pdf[0])
        for json_item in args.json:
            json_path = Path(json_item)
            if args.hub_layout or data_dir:
                local_study_dir = None
            else:
                local_study_dir = paper_local_stage4_dir(json_path, args.study_id)
            print(f"Running Stage 4: Build HumanStudy-Bench study package for {json_item}")
            summary = build_human_study_package(
                json_path,
                data_dir=data_dir or json_path.parent,
                study_dir=local_study_dir,
                study_id=args.study_id,
                pdf_path=pdf_path if pdf_path and pdf_path.exists() else None,
                provider=_llm_cfg.provider if _llm_cfg else (args.provider or "openai"),
                model=_llm_cfg.model if _llm_cfg else (args.model or "gpt-4o-mini"),
                api_key=_llm_cfg.api_key if _llm_cfg else args.api_key,
                api_base=_llm_cfg.api_base if _llm_cfg else args.api_base,
                use_llm=stage4_generation_llm_requested and _selection_llm_client is not None,
                selection_llm_client=_selection_llm_client,
                generate_config=not args.no_generate_config,
                update_registry=bool(data_dir) and not args.hub_layout,
                hub_layout=args.hub_layout,
            )
            print(
                "✓ Stage 4 complete: "
                f"study_id={summary['study_id']} json_generation={summary['json_generation']} "
                f"config={summary['config_status']}"
            )
            if summary.get("config_error"):
                print(f"  Stage 4 config error: {summary['config_error']}")
            print(f"  Study package: {summary['study_dir']}")
        return

    from generation_pipeline.pipeline import GenerationPipeline, paper_id_from_pdf
    llm_config = resolve_stage_llm_config(
        settings,
        stage=stage,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    pipeline = GenerationPipeline(
        provider=llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        output_dir=output_dir,
        settings=settings,
    )

    if stage == 3:
        stage2_path = Path(args.json[0]) if args.json else None
        pdf_path = find_pdf(args.pdf) if args.pdf else None
        if stage2_path:
            paper_dir = stage2_path.parent
        elif pdf_path:
            paper_dir = output_dir / paper_id_from_pdf(pdf_path)
        elif args.json_dir or args.pdf_dir:
            raise SystemExit(
                "Legacy Stage 3 corpus patching is not the default OSF-linker path. "
                "Run Stage 3 with --pdf <paper.pdf> or --json <paper_dir>/stage2.json."
            )
        else:
            raise SystemExit("Stage 3 requires --pdf <paper.pdf> or --json <paper_dir>/stage2.json")
        try:
            pipeline.run_stage3_paper(
                paper_dir=paper_dir,
                stage2_path=stage2_path,
                pdf_path=pdf_path,
                osf_files_dir=args.osf_dir[0] if args.osf_dir else None,
                slot_fill=args.stage3_slot_fill,
                select_studies=True,
                backup=not args.no_backup,
                write=not args.dry_run,
                allow_effect_slot_fallback=args.allow_effect_slot_fallback,
                selection_votes=args.stage3_select_votes,
                selection_timeout=args.stage3_select_timeout if args.stage3_select_timeout > 0 else None,
                pdf_material_timeout=args.stage3_pdf_timeout if args.stage3_pdf_timeout > 0 else None,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            sys.exit(1)
        return

    pdf_path = find_pdf(args.pdf)
    paper_id = paper_id_from_pdf(pdf_path)

    try:
        if stage == 1:
            if args.refine:
                review = find_latest_stage_file(1, output_dir, paper_id=paper_id).with_suffix(".md")
                if not review.exists():
                    raise FileNotFoundError(f"Review file not found: {review}")
                print(f"Refining Stage 1 from {review.name}")
                print(f"Review status: {pipeline.check_stage1_review(review)['action']}")
            pipeline.run_stage1(
                pdf_path,
                verify_inventory=not args.no_stage1_verifier,
                verifier_timeout=None if args.stage1_verifier_timeout <= 0 else args.stage1_verifier_timeout,
                auto_refine_attempts=0
                if args.no_stage1_auto_refine
                else args.stage1_auto_refine_attempts,
            )
            if args.fetch_sources:
                from generation_pipeline.connectors.registry import fetch_json_sources

                stage1_json = find_latest_stage_file(1, output_dir, paper_id=paper_id)
                results = fetch_json_sources(
                    [stage1_json],
                    pdf_paths=[pdf_path],
                    token=args.osf_token,
                    dry_run=args.dry_run,
                    extract_text=not args.no_extract_text,
                    llm_client=pipeline.client,  # use same client for link-discovery LLM fallback
                )
                print_source_fetch_results(results)
                if any(item["summary"]["errors"] for item in results):
                    sys.exit(1)

        elif stage == 2:
            regen = Path(args.regeneration_instructions) if args.regeneration_instructions else None
            if args.refine:
                review = find_latest_stage_file(2, output_dir, paper_id=paper_id).with_suffix(".md")
                if not review.exists():
                    raise FileNotFoundError(f"Review file not found: {review}")
                print(f"Refining Stage 2 from {review.name}")
                print(f"Review status: {pipeline.check_stage2_review(review)['action']}")
            stage1_json = find_latest_stage_file(1, output_dir, paper_id=paper_id)
            pipeline.run_stage2(
                stage1_json,
                pdf_path,
                regeneration_instructions_path=regen,
                grounded=bool(args.stage2_grounding and not args.no_grounding),
                ground_threshold=args.ground_threshold,
                ground_k=args.ground_k,
                ground_timeout=args.ground_timeout,
                ground_workers=args.ground_workers,
                verify_findings=not args.no_stage2_verifier,
                verifier_timeout=None if args.stage2_verifier_timeout <= 0 else args.stage2_verifier_timeout,
                auto_refine_attempts=0
                if args.no_stage2_auto_refine
                else args.stage2_auto_refine_attempts,
            )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
