## 10 February 2015

## load external libraries
library(proxy,quietly = T)
library(reshape2,quietly = T)

## get directory working address
wd <- getwd()

## load data tables
dat_mem_ord <- read.table(file = paste0(wd,"/dat_mem_ord.csv"),
                          header = T,
                          sep = ",",
                          na.strings = "NA",
                          stringsAsFactors = F)
dat_rev     <- read.table(file = paste0(wd,"/dat_rev.csv"),
                          header=T,
                          sep=",",
                          na.strings="NA",
                          stringsAsFactors = F)


## convert characters to binary
dat_mem_ord$marriage <- ifelse(dat_mem_ord$marriage == "N", 0,
                               ifelse(dat_mem_ord$marriage == "Y",1, NA) )
dat_mem_ord$gender <- ifelse(dat_mem_ord$gender == "M", 0,
                             ifelse(dat_mem_ord$gender == "W", 1, NA) )


## transform long-from reviews table to wide-form (mem_no ~ pcode)
dat_point_long <- reshape2::dcast(dat_rev[ ,c("mem_no","pcode","point")],
                                   mem_no ~ pcode,
                                   mean,
                                   value.var = "point")

## merge product reviews table with full member list
memdf <- data.frame(mem_no = order(unique(dat_mem_ord$mem_no)))
dat_point_long <- merge(x = memdf, y = dat_point_long, by = "mem_no", all = T)

## make a binary matrix for mem_no~products if product was: 1=reviewed, 0=not
dat_rev_long <- dat_point_long
dat_rev_long[is.na(dat_rev_long)] <- 0
dat_rev_long[,-1] <- ifelse(dat_rev_long[,-1] > 0, 1, 0)


## preallocate array
n <- nrow(dat_mem_ord)
x <- array(data = rep(NA,n*n*8),
           dim = c(n,n,8),
           dimnames = list(i = dat_mem_ord$mem_no,
                           j = dat_mem_ord$mem_no,
                           sim_type = c("gen_dist",
                                        "mar_dist",
                                        "ord_per_yr_dist",
                                        "avg_ord_amt_dist",
                                        "avg_yr_amt_dist",
                                        "age_dist",
                                        "rev_sim",
                                        "point_sim")
                           )
           )

## compute similarities and distances matrices
## assign  distance_ijk / MAX_ij(distance_ijk) ->  x[i,j,k]
## assign  similiarity_ijk                     ->  x[i,j,k]
## metrics:
gen_dist <- proxy::dist(dat_mem_ord$gender, diag = T, upper = F)
x[,,1] <- as.matrix(gen_dist)

mar_dist <- proxy::dist(dat_mem_ord$marriage, diag = T, upper = F)
x[,,2] <- as.matrix(mar_dist)

ord_per_yr_dist <- proxy::dist(dat_mem_ord$ord_per_yr, diag = T, upper = F)
x[,,3] <- as.matrix(ord_per_yr_dist)

avg_ord_amt_dist <- proxy::dist(dat_mem_ord$avg_ord_amt, diag = T, upper = F)
x[,,4] <- as.matrix(avg_ord_amt_dist)

avg_yr_amt_dist <- proxy::dist(dat_mem_ord$avg_yr_amt, diag = T, upper = F)
x[,,5] <- as.matrix(avg_yr_amt_dist)

age_dist <- proxy::dist(dat_mem_ord$age, diag = T, upper = F)
x[,,6] <- as.matrix(age_dist)

## similarities:
rev_sim <- proxy::simil(dat_rev_long, diag = T, upper = F)
x[,,7] <- as.matrix(rev_sim[1:(n*n)])

point_sim <- proxy::simil(dat_point_long, diag=T, upper = F)
x[,,8] <- as.matrix(point_sim[1:(n*n)])


## save sim and dist matrices as output tables
write.table(x = as.matrix(gen_dist),
            file = "gen_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(mar_dist),
            file = "mar_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(ord_per_yr_dist),
            file = "ord_per_yr_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(avg_ord_amt_dist),
            file = "avg_ord_amt_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(avg_yr_amt_dist),
            file = "avg_yr_amt_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(age_dist),
            file = "age_dist.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(rev_sim),
            file = "rev_sim.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)
write.table(x = as.matrix(point_sim),
            file = "point_sim.csv",
            sep = ",",
            na = "NA",
            col.names = TRUE,
            row.names = FALSE)

#save full array as R data types
dput(x = x, file = "sim_array.Rdata")
save(x, file = "sim_array_save.rda")
