file<-sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE))
file<-gsub("~+~"," ",file,fixed=TRUE)
source(file.path(dirname(normalizePath(file)),"statistics.R"))
d<-data.frame(condition=c("control","feedback"),cluster_id=c("one","two"),baseline_wle=c(0,NA))
for(g in c("lap","pap","pav","wav")) {
 for(i in 1:4) d[[paste0("p1_ach_",g,"_",i)]]<-c(c(1,NA,3,NA)[i],NA)
}
for(tp in c("p1","p3")) for(i in 1:4) d[[paste0(tp,"_se_w_",i)]]<-c(c(2,4,NA,NA)[i],NA)
s<-prepare_scores(d)
stopifnot(s$p1_ach_pap[1]==2,s$p1_se_w[1]==3,is.na(s$p1_ach_pap[2]),
          is.na(s$p3_se_w[2]),identical(s$fb,c(1L,2L)),is.na(s$p2_wle[2]))
cat("R scoring boundary checks passed\n")
