# GENIUS integration

This integrates contributor PR #26, preserving its original commits, study author
attribution and release documentation. The library identifier is `study_016` so
existing web and desktop catalogs can discover it without accepting arbitrary
filesystem paths. Original contribution ID: `thorbenjansen_genius_self_efficacy`.

The playground selects the study's explicit version-1 persistent protocol before
creating an independent-trial pool. It retains 23 participant administrations,
five independent feedback contexts in the feedback arm, generated baseline essays,
the frozen scorer, and source-specific evaluator. Participant and feedback model
choices are recorded. Shared-key budgets count up to 28 calls per participant;
small hosted runs are diagnostic, not full 801-participant comparisons. Progress
updates identify the current task and completed participants are checkpointed.

The contribution includes expressly authorized pseudonymous participant-level
human reference extracts and a text-derived scorer under its documented release
basis. They remain evaluator-side; do not describe the package as containing no
human data or as having received a new privacy certification. No private platform
user data is introduced by this integration.

Independent review: GPT-6.1 Terra reviewed the contribution and compatibility
diff. Its missing-statistics handling, dropped diagnostic limitations, and actual
returned-count budget findings were addressed. Offline acceptance: 208 tests pass;
the actual hosted CLI exercises both conditions through 23 stages each and returns
not_scored for developer fixtures, with no replication rate. Mock-provider tests
verify 46 participant calls, five fresh feedback calls and explicit OpenRouter
routing; these are not real model outcomes.

Still unverified: live HTTP provider behavior, full-sample R/MDMB execution,
historical feedback equivalence, and scorer validity against human ratings or
generated essays. No paid model run or desktop installer release was performed.
