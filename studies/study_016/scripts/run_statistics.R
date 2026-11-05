args <- commandArgs(trailingOnly=TRUE)
if(length(args)<3L) stop("Usage: run_statistics.R input.csv output.json goal [burnin iter nimp seed]")
script <- sub("^--file=", "", grep("^--file=",commandArgs(),value=TRUE))
script <- gsub("~+~", " ", script, fixed=TRUE)
source(file.path(dirname(normalizePath(script)),"statistics.R"))
input <- read.csv(args[1],check.names=FALSE,na.strings=c("NA",""))
burnin <- if(length(args)>=4) as.integer(args[4]) else 5000L
iter <- if(length(args)>=5) as.integer(args[5]) else 10000L
nimp <- if(length(args)>=6) as.integer(args[6]) else 20L
seed <- if(length(args)>=7) as.integer(args[7]) else 20261020L
res <- fit_self_efficacy(input,args[3],burnin,iter,nimp,seed,
                        diagnostic=burnin<5000L || iter<10000L || nimp<20L)
dir.create(dirname(args[2]),recursive=TRUE,showWarnings=FALSE)
jsonlite::write_json(res,args[2],auto_unbox=TRUE,pretty=TRUE,digits=16,na="null")
cat("Completed ",args[3],": ",res$execution_kind,"; ",res$n," records\n",sep="")
