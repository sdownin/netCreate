#-------------------
# #available tables
# sqlTables(ch)
# sqlTables(ch, tableType = "TABLE")
# sqlTables(ch, schema = "some pattern")
# sqlTables(ch, tableName = "some pattern")
# # available columns in table
# sqlColumns(ch,"table name")
# # table to dataframe
# res <- sqlFetch(ch, "table name", max = m)
# res <- sqlFetchMore(ch, "table name", max = m)
# # SQL query
# sqlQuery(ch, paste('SELECT "State", "Murder" FROM "USArrests"',
#                    'WHERE "Rape" > 30 ORDER BY "Murder"'))
# # save dataframe to table
# sqlDrop(ch, "table name", errors = FALSE)
# sqlSave(ch, some_data_frame)
# #close connection
# odbcClose(ch)


# CREATE TABLE `Cconma_Member` (
#   `mem_no` int(10) unsigned NOT NULL,
#   `name` varchar(50) NOT NULL DEFAULT '',
#   `member_birth` varchar(8) DEFAULT '' COMMENT 'date of birth',
#   `member_sex` enum('M','F') DEFAULT 'F' COMMENT 'M for man, F for woman',
#   `member_age` tinyint(4) DEFAULT NULL COMMENT 'age of member',
#   `zip` varchar(7) DEFAULT 'zipcode',
#   `addr1` varchar(255) DEFAULT 'address',
#   `gender` enum('M','W') DEFAULT 'M' COMMENT 'M for man, W for woman',
#   `birthday` varchar(10) DEFAULT NULL,
#   `f_marriage` enum('Y','N') DEFAULT 'N' COMMENT 'whether member is married or not',
#   `recommender` int(20) DEFAULT '0' COMMENT 'the mem_no who recommended me Cconma',
#   `reg_date` datetime NOT NULL ,
#
#
#
# CREATE TABLE `Cconma_Order` (
#   `ocode` varchar(20) NOT NULL DEFAULT '',
#   `seller_mem_no` int(10) unsigned NOT NULL DEFAULT 0,
#   `mem_no` int(10) unsigned NOT NULL DEFAULT '0',
#   `order_date` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
#   `amount` int(10) unsigned NOT NULL DEFAULT 0 COMMENT 'the payment amount of this order in KRW',
#
#
#
# CREATE TABLE `Cconma_Delivery` (
#   `ocode` varchar(20) DEFAULT NULL,
#   `rp_name` varchar(50) DEFAULT NULL,
#   `rp_zipcode` char(7) DEFAULT NULL,
#   `rp_addr1` varchar(100) DEFAULT NULL,
#   PRIMARY KEY (`ocode`),
#   KEY `i_rp_name` (`rp_name`)
# ) ENGINE=MyISAM;
#
#
# CREATE TABLE `Cconma_ProductReview` (
#   `serial_no` int(10) unsigned NOT NULL AUTO_INCREMENT,
#   `mem_no` int(10) unsigned NOT NULL,
#   `ocode` varchar(20) DEFAULT NULL,
#   `pcode` varchar(20) DEFAULT NULL COMMENT 'Product code',
#   `name` varchar(20) DEFAULT NULL COMMENT 'name',
#   `subject` varchar(200) DEFAULT NULL COMMENT 'the subject of product review',
#   `point` tinyint(4) DEFAULT NULL COMMENT 'product review scores from 1 to 5, the higher the better',
#   `reg_date` datetime DEFAULT NULL,
#   `ip_addr` varchar(15) DEFAULT NULL ,
#   PRIMARY KEY (`serial_no`),
#




setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\network analysis\\data")
library(sqldf)
library(sdowninMisc)
#library(stargazer)
library(plyr)
library(ggplot2)
#library(ggmap)
#library(maps)
library(lattice)
library(RODBC)
#library(igraph)
library(reshape2)
library(MASS)
library(vcd)
library(Matrix)
library(huge)
library(ff)
library(ffbase)
library(randomForest)
library(bigmemory)
library(biganalytics)
library(biglm)
library(bigalgebra)
library(lubridate)
library(proxy)
library(lmtest)
library(xts)
library(TTR)
library(memisc)
library(lme4)
source("pairs_options_function.R")
source("fitPareto_function.R")
con <- odbcConnect(dsn = "Cconma_db1",
                   uid = "root",
                   pwd = "Rebwooky2008")

# --------------------------------------------------------------
q <- '
SELECT  c.mem_no AS mem_no,
        datediff(DATE(now()),DATE(a.member_birth))/365 AS age,
        a.f_marriage AS marriage,
        a.gender AS gender,
        datediff(DATE("2014-11-07"),DATE(a.reg_date))/365 AS mem_tenure,
        Count(c.mem_no)
FROM    cconma_member AS a
JOIN    cconma_productreview AS c
ON      a.mem_no = c.mem_no
GROUP BY mem_no
HAVING   Count(c.mem_no) >= 1;
'
datall <- sqlQuery(con,q)
length(unique(datall$mem_no))




# members who have ordered >=1  ---------------------------------
q0 <- '
SELECT  a.mem_no AS mem_no,
        datediff(DATE(now()),DATE(a.member_birth))/365 AS age,
        a.f_marriage AS marriage,
        a.gender AS gender,
        datediff(DATE("2014-11-07"),DATE(a.reg_date))/365 AS mem_tenure,
        Count(b.ocode) AS ocount,
        Sum(b.amount) AS oamt_tot,
        Count(b.ocode) / (datediff(DATE("2014-11-07"),DATE(a.reg_date))/365) AS ord_per_yr,
        (Sum(b.amount) / (Max(Year(b.order_date)) - Min(Year(b.order_date)))) AS avg_yr_amt,
        Avg(b.amount) AS avg_ord_amt,
        datediff(DATE("2014-11-07"),DATE(a.last_login_time)) AS time_since_last_log
FROM    cconma_member AS a
JOIN    cconma_order AS b
ON      a.mem_no = b.mem_no
GROUP BY mem_no
HAVING   Count(b.ocode) >= 1;
'
dat0 <- sqlQuery(con,q0)
n0 <- nrow(dat0)

# Add age categorical variable
dat0$agecat <- NA
dat0$agecat[dat0$age == 0 | is.na(dat0$age)] <- "NA"
dat0$agecat[                dat0$age <= 40 ] <- "[  -40]"
dat0$agecat[dat0$age > 40 & dat0$age <= 45 ] <- "[41-45]"
dat0$agecat[dat0$age > 45 & dat0$age <= 50 ] <- "[46-50]"
dat0$agecat[dat0$age > 50 & dat0$age <= 55 ] <- "[51-55]"
dat0$agecat[dat0$age > 55                  ] <- "[56-  ]"
dat0$agecat <- as.factor(dat0$agecat)


dat03 <- subset(dat0, subset=dat0$ocount >2)
dat01 <- subset(dat0, subset=dat0$ocount >0)

#---------------------------------------------------------------------
# 1. Gender & # 2 Marital status -----------------------------------

### Density (avg order count, avg amount) by category
# avg_order_amount
plot1 <- ggplot(data = na.omit(dat01), aes(x=avg_ord_amt, fill=gender)) +
  geom_density(alpha=0.5) +
  scale_x_log10() +
  facet_wrap( ~ agecat) +
  ggtitle("Log Average Order Amount\nBy Gender and Age Category") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
ggsave("log_avg_ord_amt_gen_age.png",plot = plot1,width = 8,height = 6,units = 'in',dpi = 200)
plot1

# order_per_year
plot2 <- ggplot(data = na.omit(dat01), aes(x=ord_per_yr, fill=gender)) +
  geom_density(alpha=0.5) +
  #scale_x_log10() +
  facet_wrap( ~ agecat) +
  ggtitle("Numer of Orders Per Year\nBy Gender and Age Category") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
ggsave("avg_ord_per_yr_gen_age.png",plot = plot2,width = 8,height = 6,units = 'in',dpi = 200)
plot2


# order_per_year
plot3 <- ggplot(data = na.omit(dat01), aes(x=ocount, fill=gender)) +
  geom_density(alpha=0.5) +
  scale_x_log10() +
  facet_wrap( ~ agecat) +
  ggtitle("Log Total Orders\nBy Gender and Age Category") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
ggsave("log_or_countr_gen_age.png",plot = plot3,width = 8,height = 6,units = 'in',dpi = 200)
plot3

#total cumulative order amount
plot4 <- ggplot(data = na.omit(dat01), aes(x=oamt_tot, fill=gender)) +
  geom_density(alpha=0.5) +
  scale_x_log10() +
  facet_wrap( ~ agecat) +
  ggtitle("Log Cumulative Order Amount\nBy Gender and Age Category") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
ggsave("log_cumu_ord_amt_gen_age.png",plot = plot4,width = 8,height = 6,units = 'in',dpi = 200)
plot4

# Time since last login
plot5 <- ggplot(data = na.omit(dat01), aes(x=time_since_ll, fill=gender)) +
  geom_density(alpha=0.5) +
  #scale_x_log10() +
  facet_wrap( ~ agecat) +
  ggtitle("Time Since Last Login\nBy Gender and Age Category") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
ggsave("time_since_ll_gen_age.png",plot = plot5,width = 8,height = 6,units = 'in',dpi = 200)
plot5


### MOSAIC PLOTTING
tab1 <- table(dat0[,c(3,9,4)])
labs <- round(prop.table(tab1,margin = 1),2)
#
values <- c(tab1)
varcat1 <- unlist(dimnames(tab1)[1])
varcat2 <- unlist(dimnames(tab1)[2])
varcat3 <- unlist(dimnames(tab1)[3])
names   <- names(dimnames(tab1))
dims <- dim(tab1) #columns then rows
TABSPROPORTIONS <- structure( c(paste(labs,"\n",
                                      "(",values,")",sep="")),
                              .Dim = as.integer(dims),
                              .Dimnames = structure( list(varcat1,
                                                          varcat2,
                                                          varcat3 ),
                                                     .Names = c(names) ) ,
                              class = "table")
png("mosaic1.png",width = 8,height = 8,units = 'in',res = 200)
  mosaic(tab1,main="Customer Demographic Mosaicplot", pop=F)
  labeling_cells(text = TABSPROPORTIONS,clip_cells=F,margin = 0,)(tab1)
dev.off()

# pairs(dat0[1:1000,-1],upper.panel = panel.cor,lower.panel = panel.smooth,diag.panel = panel.hist)

# 3 Age  ----------------------------------------------

### Age plot
plot31 <- ggplot(data=na.omit(dat0), mapping=aes(x=age, fill=marriage)) +
  geom_histogram(alpha=.6,binwidth=1)  +
  ggtitle("User Distributions\nAge (NA's omitted)") +
  facet_wrap( ~ gender) +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
plot31
ggsave("age_hist_demog.png",plot = plot31,width = 8, height = 6, units = 'in', dpi = 200)

# use subset of data for 3 or more orders
# Avg yearly order amount plot
# plot32 <- ggplot(data=sample(dat01,size = 1000,replace = F),
#                  mapping=aes(x=avg_yr_amt, fill=gender)) +
#   geom_histogramm(alpha=.6,binwidth=1)  +
#   #scale_x_log10()+
#   ggtitle("User Distributions\nAverage Yearly Order Amount") +
#   facet_wrap( ~ agecat) +
#   theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
# #plot32
# ggsave("lnavg_yr_amt_hist_samp.png",plot = plot32,width = 8, height = 6, units = 'in', dpi = 200)




# 4 Geographic Proximity ----------------------------------------------



# 5 Products reviewed ----------------------------------------------
q5 <- '
SELECT  mem_no,
        pcode AS pcode,
        point AS point
FROM    cconma_productreview;
'
dat5 <- sqlQuery(con,q5)
n <- nrow(dat5)



# 6 Product preference ----------------------------------------------

# library(NMF)
nmf1 <- nmf(sampmat[,1:(ncol(sampmat)/2)],2)

ind <- sample(unique(dat5$mem_no),size = 10000,replace = F)
samp <- dat5[which(dat5$mem_no %in% ind), ]
sampmat <- dcast(samp[ ,c("mem_no","pcode","point")],mem_no ~ pcode,mean)
ratcor <- as.big.matrix(as.matrix(simil(x = sampmat,method = "euclidean",
                                        diag = T,upper = F,by_rows = T)),
                        type = "double",
                        backingfile="point_corr_mat.Rdata",
                        descriptorfile="point_corr_mat.Rdata.desc")

### 1. Create SQL table for rating point correlation
sql1 <- '
CREATE TABLE `point_corr` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `mem_no_i` int(10) unsigned NOT NULL,
  `mem_no_j` int(10) unsigned NOT NULL,
  `cor` float NOT NULL,
  PRIMARY KEY (`id`)
)
'
sqlQuery(con,sql1)

### 2. loop to compute each pairwise correlation and insert into SQL table

simSQLinsert <- function(x,              # df col order: vari,varj,val
                         vari="mem_no",    #customer (columns)
                         varj="pcode",    #product  (rows)
                         value="point"    #rating   (elements)
                         ) {
  library(reshape2)
  x[,vari] <- as.character(x[,vari])
  x[,varj] <- as.character(x[,varj])
  x[,value] <- as.numeric(x[,value])
  idvec <- unique(x[,vari])
  n <- length(idvec)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i > j) {
        pair <- c(idvec[i],idvec[j])
        # subset x only rows including pair (i,j) then dcast
        sub <- x[which(x[[vari]] %in% pair), c(vari,varj,value)]
#         sub <- sub[which(all(pair %in% sub[,varj])),]
        # sub-subset only products reviewed by both i,j
        # get() pulls the value of the variable to enter as argument
        # where the variable can't be evaluated in dcast formula.
        # The "value" variable on the other hand is just used
        # as an argument directly.
        long <- dcast(sub, get(vari) ~ get(varj), mean, value.var = value)
        rownames(long) <- long[,1]
        names(long)[1] <- "mem_no"
        corval <- cor(as.numeric(long[1,-1]),as.numeric(long[2,-1]))
        sql2 <- sprintf('INSERT INTO point_corr (mem_no_i,mem_no_j,cor) VALUES (%s,%s,%f);',pair[1],pair[2],corval)
        sqlQuery(con,sql2)
      }
    }
    if(i %% 50 == 0){cat("completed member ",i,"\n")}
  }
} # end function


long <- dcast(dat5[1:500,], mem_no ~ pcode,mean, value.var = value)
rownames(long) <- long[,1]
long <- long[,-1]

corval <- simil(long,diag = T,upper = F)


#---------------------------------------------------------------------
#
# Outcome Predictors (non-relational)
#
#---------------------------------------------------------------------

#---------------------------------------------------------------
# 1 Time between purchase -------------------------------------
#---------------------------------------------------------------
qo1 <- '
SELECT  ocode,
        mem_no,
        Date(order_date) AS date,
        amount
FROM    cconma_order;
'
dato <- sqlQuery(con,qo1)
n <- nrow(dato)


# fitting exponential distribution to time between orders ########
# took several HOURS to compute
memvec <- unique(dato$mem_no[dato$mem_no >0])
tb2 <- data.frame(mem_no=memvec,lambda=rep(NA,length(memvec)))
tb2list <- list()
  for(i in 1:length(memvec)) {
    datvec <- subset(dato,subset = (mem_no==memvec[i]),select = "date")
    # Condition on > 3 orders, else set parameter NA
    if (nrow(datvec) >3) {
      datdf <-  as.numeric(datvec[-1,1] - datvec[-nrow(datvec),1])
      tb2list[[i]] <- datdf
      tb2[i,2] <- MASS::fitdistr(x = na.omit(datdf),densfun = "exponential")$estimate
    } else {
       tb2[i,2] <- NA
       tb2list[[i]] <- NA
    }

    if(i%%100==0){cat("\ncompleted ",i," members")}
}
names(tb2list) <- memvec

dput(x = tb2list,file = "tb2list.Rdata")
dput(x = tb2, file = "tb2.Rdata")
tb2 <- dget("tb2.Rdata")


# customer-specific list to dataframe across whole platform
# tb2vec <- unlist(dget(file = "tb2list.Rdata"))
# tb2df <- data.frame(mem_no=names(tb2vec),daysBetween=tb2vec)
# write.table(x = tb2df,file = "tb2df.csv",sep = ",")
# d <- dget("tb2df.Rdat")

####


# get stored expo dist lambda values for members
library(sqldf)
file1 <- file("tb2df.csv")
tb2df2 <- read.csv.sql(file = "tb2df.csv",sql = "select * from file",header = T,row.names=T,sep = ",", stringsAsFactors = T,nrows = 10)

################################################################
# fit distribution to population ###############################
histsub <- na.omit(tb2df$daysBetween[which(tb2df$daysBetween<=1000)])
est1 <- fitdistr(x = histsub,densfun = "exponential")$est
est2 <- fitdistr(x = histsub, densfun = "cauchy")$est
est3 <- pareto.fit(data = histsub, threshold = 1)$exponent
##########

xlim2 <- 240
#plot histogram of mean times between#########
png("tb_hist_labeled.png",width = 8,height = 6,units = 'in',res = 250)
hist(na.omit(tb2df$daysBetween),
     xlab="Days",
     main="Platform Distribution of Time-Between-Orders\nby Customer (>=3 orders, NAs omitted)",
     freq = F,col="blue",
     xlim=c(0,xlim2),
     breaks = 200)
#------------------------------------------------
#### Density Curves
lines(dexp(seq(from = 0,to = xlim2,length.out = xlim2),rate = est1),lwd=3,col='red')
lines(dpareto(seq(from = 0,to = xlim2,length.out = xlim2),threshold = 1,exponent = est3),lwd=3,col='green')
#---------------------------------------------------
#### Exponential: alpha = 0.9
abline(v=qexp(p = .9,rate = est1),col='red')
text(x=qexp(p = .9,rate = est1),y=0.06,paste0("90%\nPr(X < ",round(qexp(p = .9,rate = est1),1)," days)"),cex=1.2,col='red')
#-------------------------------------------------------
#### Pareto: alpha = 0.9
pare <- qpareto(p = .9,threshold = 1,exponent = est3)
abline(v=pare,col='green')
text(x=pare,y=0.02,paste0("90%\nPr(X < ",round(pare,1)," days)"),cex=1.2,col='green')
#-------------------------------------------------------
#### legend
legtext <- c(paste0("Fitted Expo Dist\nLambda=",round(est1,3),"\n"),
                paste0("Fitted Pareto Dist\nPower=",round(est3,3),"\n"),
                "Histogram of obs")
legend(x = 'topright',legend = legtext,
       cex=1,lwd=c(2,2,10),col=c('red','green','blue')
       )
dev.off()
################################### end plot ##############

##################################
#Classify Churn Risk and Churned based on last time since last login or last purchase
qc1 <- '
SELECT  a.mem_no,
        datediff(DATE("2014-11-07"),DATE(a.last_login_time)) AS time_since_last_log,
        Count(b.ocode) AS count
FROM    cconma_member AS a
JOIN    cconma_order AS b
ON      a.mem_no = b.mem_no
GROUP BY mem_no;
'
datc <- sqlQuery(con,qc1)
n <- nrow(datc)
# tb$daysBetween <- 1/tb[,"lambda"]
# tbll <- merge(tb,datc,by = "mem_no")
# tbll$dexp <- dexp(x = tbll[,4],rate = 1/tbll[,3])
# tbll$pexp <- pexp(dexp(x = tbll[,4],rate = 1/tbll[,3]),rate = 1/tbll[,3],
#                   lower.tail = F)
# tbll$risk <- ifelse(tbll[,6]>=0.99999 & tbll[,6]<0.9999999,1,0)
# tbll$lost <- ifelse(tbll[,6]>=0.9999999,1,0)
# count(tbll[,c(7,8)])

#mem_no of members at risk
# riskmem <- tbll[which(tbll$risk==1),1]
# memvec <- unique(tbll$mem_no)

#--------------------------------------------------------------------
# #### Get full distributions time orders
# tbr <- data.frame(mem_no=memvec,lambda=rep(NA,length(memvec)))
# tbrlist <- list()
#   for(i in 1:length(memvec)) {
#     datvec <- subset(dato,subset = (mem_no==memvec[i]),select = "date")
#     # Condition on > 3 orders, else set parameter NA
#     if (nrow(datvec) >3) {
#       datdf <-  as.numeric(datvec[-1,1] - datvec[-nrow(datvec),1])
#       tbrlist[[i]] <- datdf
#       tbr[i,2] <- pareto.fit(data = na.omit(datdf),threshold = 1)$exponent
#     } else {
#       tbr[i,2] <- NA
#       tbrlist[[i]] <- NA
#     }
#
#     if(i%%100==0){cat("\ncompleted ",i," members")}
#   }
#   names(tbrlist) <- memvec
#
# dput(x = tbrlist,file = "tbrlist.Rdata")
# dput(x = tbr, file = "tbr.Rdata")
#---------------------------------------------------------------

# write.table(sep = ",",file = "tb_churn_risk.csv",
#             x = data.frame(mem_no=tbll[which(tbll$risk==1),c("mem_no")],
#                        ChurnRiskProb=tbll[which(tbll$risk==1),c("pexp")]),
#             row.names=F,col.names=T
#             )
####

# tbrlist <- dget(file = "tbrlist.Rdata")

# #Example risk probability plotting
# #remove NA
# tbllna <- na.omit(tbll)
# #order by at risk prob
# tbllna <- tbllna[order(tbllna$pexp,decreasing = F),]
# #subset by whose in the tbr list at risk
# tbllna <- tbllna[which(tbllna$mem_no %in% tbr$mem_no),]
# # one from regular, at risk, and lost
# # set.seed(115)
# # subno <- sample(x = names(tbrlist)[which(sapply(tbrlist,
# #                                                 length)>5)],
# #                 size = 4)


### CHOOSE SUBSET FOR CURN RISK GRAPHS -------------------------------------
set.seed(117)
memvecsub <- datc$mem_no[which(datc$time_since_last_log>50 & datc$count > 30)]
subno <- sample(memvecsub, 4, F)

# PLOTTING CHURN RISK GRAPHS
## Risk members #####################################
png("tb_toprisk_mem.png",width = 8,height = 8,units = 'in',res = 300)
par(mfrow=c(2,2))
par(xpd = FALSE)
for (i in 1:length(subno)) {
  m1 <- subno[i]
  est1 <- pareto.fit(data = tb2list[[which(names(tb2list)==m1)]],threshold = 1)$exponent
#   est2 <- fitdistr(x = tb2list[[which(names(tb2list)==m1)]],densfun = "exponential")$est
  hist(tb2list[[which(names(tb2list)==m1)]],
       xlab="Days",
       xlim=c(0,90),
       main=paste0("Inactivity Indication of Churn Risk\nMember ",m1),
       freq = F,col="blue",
       breaks= 30 )  #30bins
  lines(dpareto(seq(from = 0,to = 90,length.out = 90),threshold = est1, exponent = est1),lwd=2,col='green')
#   lines(dexp(seq(from = 0,to = 200,length.out = 200),rate = est2),lwd=2,col='red')
  #### actual days since last ##########
  abline(v=datc[which(datc$mem_no == m1),"time_since_last_log"],col='green')
  text(x=datc[which(datc$mem_no == m1),"time_since_last_log"],
       y=0.03,
       paste0("Last Login = ",
              round(datc[which(datc$mem_no == m1),"time_since_last_log"],1),
              " days ago\n",
              "Risk prob = ",
              round(ppareto(datc[which(datc$mem_no == m1),"time_since_last_log"],
                            threshold = 1, exponent= est1),4)
              ),
       cex=1.2,col='green')
#   #### alpha = 0.95
#   abline(v=qexp(p = .95,rate = est1))
#   text(x=qexp(p = .95,rate = est1),y=0.30,paste0("alpha[0.95] = ",round(qexp(p = .95,rate = est1),1)," days"),cex=0.9)
  #### legend
  legend(x = 'topright',legend = c(paste0("Fitted Pareto Dist\nPower=",round(est1,3),"\n"),
                                   "Histogram of observed\ndays between orders\n"),lwd=c(2,10),col=c('green','blue'))
}
dev.off()
#----------------------------------------------------------------




#-----------------------------------------------------
#
# 2 Moving Average Rating (Changing satisfaction)
#
#-------------------------------------------------------

q2 <- '
SELECT  a.ocode AS ocode,
        a.mem_no AS mem_no,
        Date(a.order_date) AS date,
        a.amount AS amount,
        b.pcode AS pcode,
        b.point AS point
FROM    cconma_order AS a
LEFT JOIN    cconma_productreview AS b
ON      a.ocode=b.ocode
UNION
SELECT  a.ocode AS ocode,
        a.mem_no AS mem_no,
        Date(a.order_date) AS date,
        a.amount AS amount,
        b.pcode AS pcode,
        b.point AS point
FROM    cconma_order AS a
RIGHT JOIN    cconma_productreview AS b
ON      a.ocode=b.ocode;
'
dat2 <- sqlQuery(con,query = q2)
n <- nrow(dat2)

## Assign unique monthly period number (based on date range)
n <- length(unique(dat2$date))
dat2$pd <- as.numeric(paste0(year(dat2$date),
                             ifelse(month(dat2$date)<10,
                                    paste0(0,month(dat2$date)),
                                    month(dat2$date)
                                    )
                             )
                      )

memvec <- unique(dat2$mem_no)
datvec <- order(unique(dat2$date))

###################################################
# Daily PLATFORM AVG time series (no demographics)
###################################################
# 1. make the period time series
x <- dat2
plat0 <- ddply(x, .(date), summarize,
              count = length(ocode),
              pointmean = mean(as.numeric(point),na.rm = T),
              pointsd = sd(as.numeric(point),na.rm = T),
              amountmean = mean(as.numeric(amount),na.rm = T),
              amountpermem = mean(as.numeric(amount),na.rm = T)/length(unique(x$mem_no)),
              amountsd = sd(as.numeric(amount),na.rm = T),
              amountsum = sum(as.numeric(amount),na.rm = F)
)

plat0[,1] <- ymd(plat0[,1])
plat <- as.xts(x = plat0[,-1],order.by = plat0[,1])
### Remove NAs individually by series and after combining to allow xts merge
y <- na.omit(cbind(na.omit(plat$pointmean),
                   na.omit(plat$amountmean/100000)))
xtsExtra::plot.xts(y["2014-05-01/"])

i <- 2 # weeks MA

yma1 <- SMA(y[,1],7*i)
yma2 <- SMA(y[,2],7*i)
yma <- merge(yma1,yma2)

xtsExtra::plot.xts(yma["2014-05-01/"],legend.loc=7)

pairs(labels = names(plat),
      na.omit(cbind(as.numeric(plat[,1]),
            as.numeric(plat[,2]),
            as.numeric(plat[,3]),
            as.numeric(plat[,4]),
            as.numeric(plat[,5]),
            as.numeric(plat[,6]),
            as.numeric(plat[,7]))),
      lower.panel=panel.smooth,upper.panel=panel.cor,diag.panel=panel.hist)

#----------------------------
##### Times series plot NO filter
# y$amtmean10000 <- y$amountmean / 10000

# Daily Averages baseline time series plot #################
# ya <- y$amtmean10000["2014-05-01/"]
# yp <- y$pointmean["2014-05-01/"]
# png(paste0("avg_amount_point.png"),width = 6,height = 6,units = 'in',res = 300)
#   plot(cbind(ya,yp),
#      main=paste0("Avg Order Value vs Avg Point"),
#      legend.loc=7)
# dev.off()
gctest <- lmtest::grangertest(x = cbind(yma[,2],yma[,1]),order = 1)
gctest

#------------------------------
#### Moving averages 1 to 5 weeks ###################
# for loop not working  DON'T know why
gctlist <- list()
  ya <- TTR::SMA(x = y$amtmean10000["2014-05-01/"],n = 7*i)
  yp <- TTR::SMA(x = y$pointmean["2014-05-01/"],n = 7*i)
  png(paste0("ma_amount_point_",i,".png"),width = 6,height = 6,units = 'in',res = 300)
      plot(cbind(ya,yp),
           main=paste0("Moving Avg Order Value vs Avg Point\n",i,
                       " Weeks"),
           legend.loc=7)
  dev.off()
  gctlist[[i]] <- lmtest::grangertest(x = cbind(yp,ya),order=1)


###################################################
# Daily PLATFORM AVG time series WITH Demographics
###################################################

####################################
# 1. SQL squery to q0, and to q2 above
# 2. Merge q0 user demograhpics into q2 amount and point
# 3. ddply splite/apply/comb on demograhpic groups
# 4. plot, MA, Granger test
####################################

dat <- merge(x = dat2,y = dat0,by = "mem_no",all = T)

#
#
#
#
#
#                DO THIS FIRST:
#
#
#
#
#
# Long Time to Split/Apply/Combine
plat <- ddply(dat, .(date,agecat), summarize,
              count = length(ocode),
              pointmean = mean(as.numeric(point),na.rm = T),
              pointsd = sd(as.numeric(point),na.rm = T),
              amountmean = mean(as.numeric(amount),na.rm = T),
              amountpermem = mean(as.numeric(amount),na.rm = T)/length(unique(dat$mem_no)),
              amountsd = sd(as.numeric(amount),na.rm = T),
              amountsum = sum(as.numeric(amount),na.rm = F)
)
# dput(x = plat, file = "daily_avg_amt_point_agecat.Rdata")
# plat <- dget("daily_avg_amt_point_agecat.Rdata")

# individuals being averaged
length(unique(na.omit(dat)$mem_no))



descplat3 <- describe(na.omit(plat[,c("pointmean","amountmean","amountsum","date","agecat")]))
write.table(x = descplat3,file = "descplat3.csv",append = F,sep = ",",col.names = T,row.names=T)
dat6$recout <- as.factor(dat6$recout)
dat6$recin <- as.factor(dat6$recin)
# ##### REMOVE OUTLIER AGE OBSERVATIONS ######################
# iqr <- range(quantile(na.omit(dat6$age),probs = c(.25,.75)))
# olr <- c(quantile(na.omit(dat6$age),probs = c(.5))-iqr[1]*1.5,
#          quantile(na.omit(dat6$age),probs = c(.5))+iqr[2]*1.5)
dat6trim <- dat6[which(dat6$age >= 16 ),]
descdat6 <- describe(dat6trim)
write.table(x = descdat6,file = "descdat6_2.csv",append = F,sep = ",",col.names = T,row.names=T)

#----------------------------------------------
# By AGE CATEGORY PLOTTING AND ANOVA

# ADD PERIOD Variable instead of DATE for REGRESSION
# only for na.omit(plat)
platna <- na.omit(plat)
pd <- data.frame(date = unique(platna$date),
                 pd = seq_len(length(unique(platna$date)))
                 )
platna <- merge(platna,pd,by="date",all=T)

##### 1. Amount Mean
gp1 <- ggplot(data = na.omit(plat), aes(x=date,colour=agecat)) +
  geom_line(aes(y=amountmean),alpha=0.8,lwd=0.5) +
  geom_smooth(data = na.omit(plat), aes(y = amountmean, colour=agecat),method=lm,se=F,lwd=2) +
  geom_point(data = na.omit(plat), aes(y = amountmean,colour=agecat)) +
  ggtitle("Avg Daily Amount by Age Category\n(2014)") +
  ylab("KRW") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),
        legend.position='right',
        panel.background = element_rect(fill = 'white', colour = 'gray'))
gp1
ggsave(filename = "avg_daily_amt_agecat_2.png",plot = gp1,width = 10,height = 5,units = 'in', dpi = 250)

aov1 <- aov(formula = amountmean ~ agecat,data = na.omit(plat))
summary(aov1)

lm1 <- lm(formula = amountmean ~ pd + agecat ,data = platna)
summary(lm1)

##### 2. Amount sum
gp2 <- ggplot(data = na.omit(plat), aes(x=date,colour=agecat)) +
  geom_line(aes(y=amountsum),alpha=0.8,lwd=0.5) +
  geom_smooth(data = na.omit(plat), aes(y = amountsum, colour=agecat),method=lm,se=F,lwd=2) +
  geom_point(data = na.omit(plat), aes(y = amountsum,colour=agecat)) +
  ggtitle("Daily Amount Sum by Age Category\n(2014)") +
  ylab("KRW") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),
        legend.position='right',
        panel.background = element_rect(fill = 'white', colour = 'gray'))
gp2
ggsave(filename = "avg_daily_amt_sum_agecat_2.png",plot = gp2,width = 10,height = 5,units = 'in', dpi = 250)

aov2 <- aov(formula = amountsum ~ agecat, data = na.omit(plat))
summary(aov2)

lm2 <- lm(formula = amountsum ~ pd + agecat ,data = platna)
summary(lm2)


##### 3. Point Mean
gp3 <- ggplot(data = na.omit(plat), aes(x=date,colour=agecat)) +
  geom_line(aes(y=pointmean,colour = agecat),alpha=0.8,lwd=0.5) +
  geom_smooth(data = na.omit(plat), aes(y = pointmean, colour=agecat),
              alpha=0.8,method=lm,se=F,lwd=1.5) +
  geom_point(data = na.omit(plat), aes(y = pointmean,colour=agecat),alpha=0.8) +
  ylab("points") +
  ggtitle("Avg Daily Product Rating by Age Category\n(2014)") +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),
        legend.position='right',
        panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid = element_line(colour = "black"))
gp3
ggsave(filename = "avg_daily_prod_rating_agecat_2.png",plot = gp3,width = 10,height = 4,units = 'in', dpi = 250)

aov3 <- aov(formula = pointmean ~ agecat + date:agecat ,data = na.omit(plat))
summary(aov3)

lm3 <- lm(formula = pointmean ~ pd + agecat + pd:agecat ,data = platna)
summary(lm3)



# ##### 4. amountpermem
# gp4 <- ggplot(data = na.omit(plat), aes(x=date,colour=agecat)) +
#   geom_line(aes(y=amountpermem),alpha=0.8,lwd=1.1) +
#   ggtitle("Avg Amount per Customer by Age Category\n(2014)")
# ggsave(filename = "avg_amt_per_cust_agecat.png",plot = gp4,width = 10,height = 5,units = 'in', dpi = 250)
#
# aov4 <- aov(formula = amountpermem ~ agecat + date,data = na.omit(plat))
# summary(aov4)
#
# lm4 <- lm(formula = amountpermem ~ agecat + date,data = na.omit(plat))
# summary(lm4)
# plot(lm4)

mtable1 <- mtable(lm1,lm2,lm3)
#----------------------------------------------


#------------------------------------------------------------------
#
# ML Classification of Recommender / Recommended
#
#------------------------------------------------------------------

q6a <- '
SELECT  a.mem_no AS mem_no,
        a.recommender AS recommender,
datediff(DATE(now()),DATE(a.member_birth))/365 AS age,
a.f_marriage AS marriage,
a.gender AS gender,
datediff(DATE("2014-11-07"),DATE(a.reg_date))/365 AS mem_tenure,
Count(b.ocode) AS ocount,
Avg(b.amount) AS avg_amt,
(Sum(b.amount) / (Max(Year(b.order_date)) - Min(Year(b.order_date)))) AS avg_yr_amt,
datediff(DATE("2014-11-07"),DATE(a.last_login_time)) AS time_since_last_log
FROM    cconma_member AS a
JOIN    cconma_order AS b
ON      a.mem_no = b.mem_no
GROUP BY mem_no;
'
dat6a <- sqlQuery(con,q6a)
n6a <- nrow(dat6a)

q6b <- '
SELECT  c.mem_no AS mem_no,
Avg(c.point) AS avg_point,
Count(c.point) AS point_count
FROM    cconma_productreview AS c
GROUP BY c.mem_no;
'
dat6b <- sqlQuery(con,q6b)
n6b <- nrow(dat6b)

dat6 <- merge(x=dat6a, y=dat6b, by="mem_no",all=T)

# if point_count is NA assign 0 reviews made
is.na(dat6$point_count) <- 0

# Add age categorical variable
dat6$agecat <- NA
dat6$agecat[dat6$age == 0 | is.na(dat6$age)] <- "NA"
dat6$agecat[                dat6$age <= 40 ] <- "[  -40]"
dat6$agecat[dat6$age > 40 & dat6$age <= 45 ] <- "[41-45]"
dat6$agecat[dat6$age > 45 & dat6$age <= 50 ] <- "[46-50]"
dat6$agecat[dat6$age > 50 & dat6$age <= 55 ] <- "[51-55]"
dat6$agecat[dat6$age > 55                  ] <- "[56-  ]"
dat6$agecat <- as.factor(dat6$agecat)




### ASSIGN RECOMMENDER CLASSES

# assign RECIN:  RECOMMENDED BY SOMEONE ELSE binary variable
is.na(dat6$recommender) <- 0
dat6$recin <- 0
dat6[which(dat6$recommender != 0 ),]$recin <- 1
# assign RECOUT:  RECOMMENDER of SOMEONE ELSE outgoing binary variable
recmemvec <- unique(dat6$recommender)
dat6$recout <- 0
dat6[which(dat6$mem_no %in% recmemvec),]$recout <- 1
# assign merge RECCOUNT variable
reccount <- count(dat6$recommender)
reccount <- reccount[-1,]    # drop mem_no 0
names(reccount) <- c("mem_no","reccount")
dat6 <- merge(dat6,reccount,by="mem_no",all=T)
#### SAVE TABLE
# dropped mem_no 0
dat60 <- dat6[-1,]
# write.table(x = dat6,file = "dat6_all.csv",sep=",",append = F,na = "NA",col.names=T,row.names=F)

rectable <- ddply(.data = dat6, .variables = c("recout","recin"),summarize,
                  meanamount = mean(avg_amt,na.rm = T),
                  meanpoint = mean(avg_point,na.rm=T))


#### add mean time between orders to dat6
tb2list <- dget("tb2list.Rdata")
ocount <- ldply(.data = tb2list, .fun = c(length,mean))
names(ocount) <- c("mem_no","ocount","mean_days_between")

dat6 <- merge(dat6,ocount,by="mem_no",all=T)


# write.table(x = dat6,file = "dat6.csv",append = F,sep = ",",row.names = F,col.names = T)


library(psych)
descdat6 <- describe(dat6)
View(descdat6)
write.table(x = descdat6,file = "descdat6.csv",append = F,sep = ",",row.names = F,col.names = T)

##### GGGPLOT BY recommender value class
dat6[,c("recout")] <- as.factor(dat6[,c("recout")])
dat6[,c("recin")] <- as.factor(dat6[,c("recin")])
gp61 <- ggplot(data = dat6, aes(x=na.omit(dat6$avg_amt), fill=recout)) +
  geom_density(alpha=0.5) +
  facet_wrap( ~ na.omit(dat6$agecat)) +
  ggtitle(paste0("Avg Amt","Density\nby Recommender Status and Age Category")) +
  theme(plot.title = element_text(lineheight=1.1, size=16, face="bold"),legend.position='bottom')
gp61
ggsave(paste0("rec_dist",i,".png"),plot = gp61,width = 6,height = 6,units = 'in',dpi = 250)


mosaic(formula= recout ~ recin + agecat,data = dat6)



##### plot of POISSON DISTRIBUTION OF RECOMMENDATION COUNT
c <- 2
png("rec_count_dist_2.png",height=8,width=6,units='in',res=250)
hist(dat6$reccount[which(dat6$reccount>=c)],freq = F,xlim=c(0,100),breaks=50,col="light blue",xlab="Recommendation Count",main="Distribution of Number of Recommendations\nto New Users Made per Current User (>=2)")
lambda <- fitdistr(x = dat6$reccount[which(dat6$reccount>=c)],densfun = "poisson")$est
lines(dpois(seq(0,100,by = 1),lambda = lambda,log = F),col='red',lwd=2)
dev.off()

# library(xts)
# plat[,1] <- ymd(plat[,1])
# plat <- as.xts(x = plat[,-1],order.by = plat[,1])
# ### Remove NA by index for XTS object ##########
# plat <- plat[which(!is.na(index(plat))),]
#
# xtsExtra::plot.xts(plat$amountmean)



#----------------------------
# #2. Filter pd time series to moving average
# y$amtmean10000 <- y$amountmean / 10000
#
# # Daily Averages baseline time series plot #################
# ya <- y$amtmean10000["2014-05-01/"]
# yp <- y$pointmean["2014-05-01/"]
# png(paste0("avg_amount_point.png"),width = 6,height = 6,units = 'in',res = 300)
# plot(cbind(ya,yp),
#      main=paste0("Avg Order Value vs Avg Point"),
#      legend.loc=7)
# dev.off()
# gctest <- lmtest::grangertest(y = ya,x = yp,order = 1)









# ## Plot moving average point vs moving average order amount
# df <- as.ts(x = df[,c(2,3)], index=df[,1])
# matplot(cbind(df[,c(1)],df[,c(2)]/10000),type='l',lty=c(1,1),ylim=c(1,5))
# df[,2] <- df[,2]/10000
# png("platform_amount_point_ma_naomit.png",width = 6,height = 6,units = 'in',res=200)
# ts.plot(na.omit(df),col=c('blue','red'),lwd=2,ylim=c(3.9,5),main="Daily Average Order Amount CumuAvg\nVsProduct Review CumuAvg",ylab="Rating; (Avg Amount/10000)",xlab="Day")
# legend(x=900,y=4.5,legend=c("PointMA","AmountMA/10000"),lty=c(1,1),lwd=c(2,2),col=c('blue','red'))
# dev.off()




#------------------------------------------------------------
#  ratings SVD  $ Demand-Utility regression
#------------------------------------------------------------

# load ratings data and
rat <- read.csv.sql(file = "dat_rev.csv",sql = "SELECT * FROM file",header=T,sep=",",row.names=T)
r <- rat[,c("mem_no","point")]
#count of raters in top 200
memfreq <- count(r$mem_no)
memfreq <- memfreq[order(memfreq[,2],decreasing = T),]
mfsub <- memfreq[2:2001,1]  #dropping mem_no 0
# ratings subset by top 200 raters
rmsub <- rat[mfsub,]

#dcast ratings matrix
rm <- as.matrix(dcast(rmsub[ ,c("mem_no","pcode","point")],
            mem_no ~ pcode,
            mean))
        # r <- as.matrix(dcast(rmsub[ ,c("mem_no","pcode","point")],
        #                      mem_no ~ pcode,
        #                      mean))
        # row.names(r) <- r[,1]
        # r <- r[-1,-1]
#assign first column to row names and then remove from matrix
row.names(rm) <- rm[,1]
rm <- rm[-1,-1]  # drop mem_no 0 fow, and mem_no column
#impute missing values
rm <- e1071::impute(x = rm, what = "mean")
##### for now just scale and assign zero
rmc <- scale(x = rm,center=T,scale=T)
#####
##### Either replace with 0
rmc[is.na(rmc)] <- 0
# ##### or just remove NA
# rmc <- na.omit(rmc)

ei <- eigen(rmc %*% t(rmc))

## CHOOSE NUMBER OF singular vectors to calculate
## and number of singular values to use
n <- 4

# SVD
rmsvd <- svd(x = rmc, nu = n,nv = n)
sigma <- rmsvd$d[1:n]
# d <- eigen(rmc %*% t(rmc))
U <- rmsvd$u[,1:n]
Vt <- t(rmsvd$v)[1:n,]

## Compute Theta for given number of eigenvalues/eigenvectors
theta <- U %*% Vt
# names cols and rows of newly calculated theta
colnames(theta) <-colnames(rmc)
rownames(theta) <- rownames(rmc)
# LONG THETA for regression
theta_l <- melt(data = t(theta))
theta_l <- theta_l[,-1]
names(theta_l) <- c("mem_no","qi")

###-------------------------------------
### import member info and run regression
df <- read.table(file = "dat_mem_ord.csv",header = T,sep = ",",na.strings = "NA")
df$marriage <- as.factor(df$marriage)
df$gender <- as.factor(df$gender)
dfsub <- subset(df,subset=(df$mem_no %in% theta_l$mem_no))
# data frame
x <- merge(theta_l,dfsub,by="mem_no",all = T)
x$mem_no <- as.factor(x$mem_no)

# Remove NAs
xna <- na.omit(x)

# simple linear regression
lm1 <- lm(xna$avg_yr_amt ~ xna$qi)

# multiple linear regression
lm2 <- lm(xna$avg_yr_amt ~ xna$qi + xna$age + xna$marriage + xna$gender)

# full predictors multiple linear regression
lm3 <- lm(xna$avg_yr_amt ~ xna$qi + xna$age + xna$marriage+ xna$gender + xna$mem_tenure + xna$ocount + xna$oamt_tot + xna$ord_per_yr + xna$avg_ord_amt + xna$time_since_last_log)

###_---------------------------------------------------

###---------------------------------------------------
### random and mixed effects
###---------------------------------------------------

# Random intercept
## Negative eigenvalues needs to be addressed
lmerControl <- lmerControl(optCtrl=list(maxfun=50000) )
xna <- na.omit(x)
# scale xna to correct negative eigenvalues in hessian
xs <- scale(xna[,-c(1,4,5)], center=T,  scale=T)
xsb <- cbind(xna[,c(1,4,5)],xs)

# simple random slope
lmer1 <- lmer(xsb$avg_yr_amt ~ xsb$qi + (xsb$qi|xsb$mem_no), control=lmerControl)

# mixed effects multiple regression
lmer2 <- lmer(xsb$avg_yr_amt ~ xsb$qi + xsb$age + xsb$marriage + xsb$gender + (1|xsb$mem_no) + (xsb$qi|xsb$mem_no), control=lmerControl)

#----------------------------------------------
# problem still have non-positive-definite VtV matrix






#--------------------------------------------------
#
# Fixed effects regression Mean rating by member
#
#-----------------------------------------------
setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\network analysis\\data")
load("cconma_ratings_svd_regression.Rdata")
save.image("cconma_ratings_svd_regression.Rdata")
#use ratings matrix rm from above but remove rows with all NA
nacols <- ifelse(colSums(is.na(rm)) > 1, F, T)
rmna <- rm[,nacols]

#NMF with given number of factors
library(NMF)
nmf2 <- NMF::nmf(x = rmna, rank = 2, seed=1)
nmf3 <- NMF::nmf(x = rmna, rank = 3, seed=1)
nmf4 <- NMF::nmf(x = rmna, rank = 4, seed=1)

estim.nmf <- NMF::nmf(x=rmna, rank = 1:6, nrun = 10, seed=1)
png('estim_nmf.png',height=8,width=10,units='in',res=250)
  plot(estim.nmf)
dev.off()
consensusmap(estim.nmf)

# # plot for 2 factors
# png("nmf3_basis.png",height=20,width=10,unit='in',res=300)
#   basismap(nmf3)
# dev.off()
# png("nmf3_coef.png",height=10,width=20,unit='in',res=300)
#   coefmap(nmf3)
# dev.off()

# U %*% V'
theta <- basis(nmf2) %*% coef(nmf2)
# average theta per member
theta_mean <- as.data.frame(apply(X = theta, MARGIN = 1, FUN = mean))
names(theta_mean) <- "theta_mean"
theta_mean$mem_no <- rownames(theta_mean)
theta_mean_na <- na.omit(theta_mean)

x <- merge(theta_mean_na,dfsub,by="mem_no",all = T)

fitall <- lm(log(x$avg_yr_amt) ~ x$theta_mean + x$age + x$marriage+ x$gender + x$mem_tenure + x$ocount + x$oamt_tot + x$ord_per_yr + x$avg_ord_amt + x$time_since_last_log)


#------------- plot ------------------------
#
#
# Add age categorical variable
x <- na.omit(x)
x$agecat[             x$age <= 40 ] <- "[  -40]"
x$agecat[x$age > 40 & x$age <= 45 ] <- "[41-45]"
x$agecat[x$age > 45 & x$age <= 50 ] <- "[46-50]"
x$agecat[x$age > 50 & x$age <= 55 ] <- "[51-55]"
x$agecat[x$age > 55               ] <- "[56-  ]"
x$agecat <- as.factor(x$agecat)


# fit1 <- lm(x$avg_yr_amt ~ x$theta_mean)
# fit2 <- lm(x$avg_yr_amt ~ x$theta_mean + x$age)
# fit3 <- lm(x$avg_yr_amt ~ x$theta_mean + x$age + x$marriage)
# fit4 <- lm(x$avg_yr_amt ~ x$theta_mean + x$age + x$marriage + x$gender)
# fit5 <- lm(x$avg_yr_amt ~ x$theta_mean + x$age + x$marriage + x$gender+ x$mem_tenure)

fit5log <- lm(log(x$avg_yr_amt) ~ x$theta_mean + x$age + x$marriage + x$gender+ x$mem_tenure)

#------------------------------------------
#  facet plot of demand by satisfaction
g5 <- ggplot(aes(y=log(avg_yr_amt), x=theta_mean + mem_tenure, colour=gender), data=x)
g5 <- g5 + geom_point(aes(colour=gender), data=x)
g5 <- g5 + geom_smooth(aes(colour=gender),
                       method=lm,se=F,lwd=1.5,data=x)
g5 <- g5 + facet_wrap(~ agecat)
g5 <- g5 + ggtitle("Demand as a Function of Satisfaction + Tenure")
g5
ggsave("demand_reg_facet_scatter.png",height=6,width=10,units='in',dpi=200)
#---------------------------------------------


#----------------------------------------------------------
# SQL query ratings join producers of products rated
# to average customer ratings by producer
# limiting columns in ratings matrix
#----------------------------------------------------------

# user and seller query
q9a <- '
SELECT  a.mem_no AS mem_no,
datediff(DATE(now()),DATE(a.member_birth))/365 AS age,
a.f_marriage AS marriage,
a.gender AS gender,
datediff(DATE("2014-11-07"),DATE(a.reg_date))/365 AS mem_tenure,
(Sum(b.amount) / (Max(Year(b.order_date)) - Min(Year(b.order_date)))) AS avg_yr_amt,
datediff(DATE("2014-11-07"),DATE(a.last_login_time)) AS time_since_last_log,
b.seller_mem_no AS seller_no
FROM    cconma_member AS a
JOIN    cconma_order AS b
ON      a.mem_no = b.mem_no
GROUP BY mem_no
HAVING   Count(b.ocode) >= 1;
'
q9b <- '
SELECT  a.mem_no,
a.pcode AS pcode,
a.point AS point,
b.seller_mem_no AS seller_no
FROM    a.cconma_productreview
JOIN    b.cconma_order
ON      a.pcode=b.pcode;
'

dat9a <- sqlQuery(con,q9a)
dat9b <- sqlQuery(con,q9b)





# replace NAs with average over the rows
rm2 <- rm
for (i in seq_len(nrow(rm2))) {
  rm2[i,is.na(rm2[i,])] <- mean(rm2[i,!is.na(rm2[i,])])
}







#---------------------------------------------
#subset of customers and recommendations with less NA
#---------------------------------------------
rowna <- rowSums(is.na(r))
colna <- colSums(is.na(r))


#----------------------------------------------------------
# 3d plotting
#---------------------------------------------------------
# scatter3d(x$theta_mean,x$mem_tenure,log(x$avg_yr_amt))
# plot3d(x$theta_mean,x$mem_tenure,log(x$avg_yr_amt))
#
# fit5sub <- lm(log(x$avg_yr_amt) ~ x$theta_mean + x$mem_tenure)
# ax1 <- seq(from=min(x$theta_mean),to=max(x$theta_mean),length.out = nrow(x))
# ax2 <- seq(from=min(x$mem_tenure),to=max(x$mem_tenure),length.out = nrow(x))
#
# #----------------------------------------------------
# b0 <- fit5sub$coefficients[1]
# b1 <- fit5sub$coefficients[2]
# b2 <- fit5sub$coefficients[3]
#
# # best 3d linear fit---------------------------
# zpred <- b0 + b1*ax1 + b2*ax2
# plot3d(c(x$theta_mean,ax1),
#        c(x$mem_tenure,ax2),
#        c(log(x$avg_yr_amt),zpred))
#
# # 3d contour---------------------------------
# z <- matrix(rep(NA,length(ax1)^2),nrow=length(ax1))
# zvec <- c()
# for (i in seq_len(nrow(z))) {
#   for (j in seq_len(ncol(z))) {
#     z[i,j] <- b0 + b1*ax1[i] + b2*ax2[j]
#     z[i,j] <- log(z[i,j])
#     zvec[(i-1)+j] <- b0 + b1*ax1[i] + b2*ax2[j]
#     zvec[(i-1)+j] <- log(zvec[(i-1)+j])
#   }
# }
# contour(ax1,ax2,z)
# plot3d(predict(fit5sub,data.frame(ax1,ax2)))


