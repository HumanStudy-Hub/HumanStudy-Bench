# Paired complete-case sensitivity, not a repeat of the published MI analysis.
args <- commandArgs(TRUE)
stopifnot(length(args)==4)
library(lmerTest)
library(clubSandwich)
d <- read.csv(args[1],check.names=FALSE,na.strings=c("NA",""))
d <- d[,nzchar(names(d)),drop=FALSE]
s <- readRDS(args[2])
raw_class <- s$token[match(d$id,s$id)]
keep <- !is.na(d$p2_wle) & !is.na(raw_class) & !is.na(d$p2_draft) & nzchar(trimws(d$p2_draft))
d <- d[keep,,drop=FALSE]
d$class <- as.factor(raw_class[keep])
p <- read.csv(args[3])
stopifnot(nrow(p)==nrow(d),max(abs(p$reference_wle-d$p2_wle))<1e-12)
d$surrogate_wle <- p$text_prediction
d$fb <- d$feedback_binaer+1
d$se_pre <- rowMeans(d[paste0("p1_se_w_",1:4)],na.rm=TRUE)
d$se_post <- rowMeans(d[paste0("p3_se_w_",1:4)],na.rm=TRUE)
results <- list()
for(goal in c("lap","pap","pav","wav")) {
  d$goal <- rowMeans(d[paste0("p1_ach_",goal,"_",1:4)],na.rm=TRUE)
  x <- d[complete.cases(d[c("p2_wle","surrogate_wle","se_pre","se_post","goal","class","fb")]),]
  x$goal <- x$goal-mean(x$goal)
  x$se_pre <- x$se_pre-mean(x$se_pre)
  x$se_post <- x$se_post-mean(x$se_post)
  for(version in c("reference","surrogate")) {
    x$w <- if(version=="reference") x$p2_wle else x$surrogate_wle
    x$w <- x$w-mean(x$w)
    m <- lmer(se_post ~ se_pre + fb*w*goal + (1|class),data=x)
    co <- coef(summary(m))
    cr <- as.data.frame(coef_test(m,vcov="CR2",cluster=x$class))
    for(term in c("fb","fb:goal","fb:w:goal")) {
      results[[length(results)+1]] <- list(goal=goal,version=version,term=term,
        n=nrow(x),classes=nlevels(droplevels(x$class)),B=unname(co[term,"Estimate"]),
        SE=unname(co[term,"Std. Error"]),p_primary=unname(co[term,"Pr(>|t|)"]),
        p_cr2=cr[term,"p_Satt"],singular=lme4::isSingular(m))
    }
  }
}
out <- list(scope="Paired complete-case random-intercept/CR2 sensitivity on 757 scored, known-class students. No imputation; NOT a published-table reproduction.",
  outcomes_used_only_for_downstream_audit=TRUE,outcomes_used_to_train_scorer=FALSE,
  results=results)
jsonlite::write_json(out,args[4],pretty=TRUE,auto_unbox=TRUE,na="null",digits=16)
