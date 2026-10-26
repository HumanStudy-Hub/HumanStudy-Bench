# Adapted from Jansen et al. (2026), OSF nte2h/analysis_mi.R.
# Raw-scale means, substantive-model-compatible MDMB imputation, random-
# intercept outcome model and separate CR2 sensitivity. No text/keystroke score.
prepare_scores <- function(items) {
  goal_codes <- c("lap", "pap", "pav", "wav")
  out <- data.frame(
    id = seq_len(nrow(items)),
    token = as.integer(factor(items$cluster_id)),
    fb = ifelse(items$condition == "feedback", 2L,
                ifelse(items$condition == "control", 1L, NA_integer_)),
    p2_wle = items$baseline_wle
  )
  if (anyNA(out$fb) || anyNA(out$token)) stop("Invalid condition or cluster")
  for (g in goal_codes) {
    cols <- paste0("p1_ach_", g, "_", 1:4)
    if (!all(cols %in% names(items))) stop("Missing goal items: ", g)
    out[[paste0("p1_ach_", g)]] <- rowMeans(items[cols], na.rm=TRUE)
  }
  for (tp in c("p1", "p3")) {
    cols <- paste0(tp, "_se_w_", 1:4)
    if (!all(cols %in% names(items))) stop("Missing self-efficacy items: ", tp)
    out[[paste0(tp, "_se_w")]] <- rowMeans(items[cols], na.rm=TRUE)
  }
  out[] <- lapply(out, function(x) {x[is.nan(x)] <- NA; x})
  out
}

imputation_models <- function() {
  dep <- list(model="mlreg", formula=p3_se_w ~
    1 + I(scale(p2_wle, scale=FALSE)[,1]) +
    I(scale(mod2_var, scale=FALSE)[,1]) +
    I(scale(p1_se_w, scale=FALSE)[,1]) +
    I(scale(p2_wle, scale=FALSE)[,1]):I(scale(mod2_var, scale=FALSE)[,1]) +
    fb:I(scale(p2_wle, scale=FALSE)[,1]) +
    fb:I(scale(mod2_var, scale=FALSE)[,1]) +
    fb:I(scale(p2_wle, scale=FALSE)[,1]):I(scale(mod2_var, scale=FALSE)[,1]) +
    fb + (1 + I(scale(p2_wle, scale=FALSE)[,1]) +
                I(scale(mod2_var, scale=FALSE)[,1]) +
                I(scale(p1_se_w, scale=FALSE)[,1]) | token),
    sampling_level="token")
  ind <- list(
    p2_wle=list(model="mlreg", formula=p2_wle ~ 1 + (1|token),sampling_level="token"),
    mod2_var=list(model="mlreg", formula=mod2_var ~ 1 + (1|token),sampling_level="token"),
    se_mod=list(model="mlreg", formula=p1_se_w ~ 1 + (1|token),sampling_level="token")
  )
  # se_mod is the source script's exact key, not a new imputation method.
  list(dep=dep, ind=ind)
}

fit_self_efficacy <- function(items, goal, burnin=5000L, iter=10000L,
                              nimp=20L, seed=20261020L, diagnostic=FALSE) {
  stopifnot(goal %in% c("lap","pap","pav","wav"))
  needed <- c("mdmb","mitml","lme4","parameters","clubSandwich")
  for (p in needed) if (!requireNamespace(p,quietly=TRUE)) stop("Install R package: ",p)
  d <- prepare_scores(items)
  d$mod2_var <- d[[paste0("p1_ach_",goal)]]
  d <- d[c("id","token","fb","p2_wle","p1_se_w","p3_se_w","mod2_var")]
  for (v in c("fb","p2_wle","p1_se_w","p3_se_w","mod2_var")) {
    if (length(unique(na.omit(d[[v]]))) < 2) stop("Insufficient variation: ",v)
  }
  if (length(unique(d$token))<3) stop("Too few independent clusters")
  mods <- imputation_models()
  set.seed(seed)
  imp <- mdmb::frm_fb(d, mods$dep, mods$ind, burnin=burnin, iter=iter,
                      Nimp=nimp, aggregation=TRUE, verbose=FALSE)
  dl <- mitml::as.mitml.list(mdmb::frm2datlist(imp))
  # Within each imputation: grand-mean centering, not classroom centering.
  for (i in seq_along(dl)) {
    for (v in c("p2_wle","mod2_var","p1_se_w","p3_se_w")) {
      dl[[i]][[paste0(v,"_cgm")]] <- as.numeric(scale(dl[[i]][[v]],scale=FALSE))
    }
  }
  models <- with(dl,lme4::lmer(p3_se_w_cgm ~
    1 + p1_se_w_cgm + p2_wle_cgm + mod2_var_cgm +
    p2_wle_cgm:mod2_var_cgm + fb + fb:p2_wle_cgm +
    fb:mod2_var_cgm + fb:p2_wle_cgm:mod2_var_cgm + (1|token)))
  primary <- as.data.frame(parameters::pool_parameters(models))
  robust <- as.data.frame(parameters::pool_parameters(models,vcov="CR2",
                             vcov_args=list(cluster=d$token)))
  list(goal=goal, n=nrow(d), clusters=length(unique(d$token)),
       seed=seed, burnin=burnin, iterations=iter, imputations=length(dl),
       execution_kind=if(diagnostic) "diagnostic_only" else "source_method_reexecution",
       source_method="OSF analysis_mi.R MDMB + random intercept; CR2 separate",
       original_publication_seed="not reported in source MDMB block",
       primary=primary, cr2=robust,
       singular_fits=sum(vapply(models,lme4::isSingular,logical(1))),
       packages=setNames(lapply(needed,function(p) as.character(utils::packageVersion(p))),needed))
}
