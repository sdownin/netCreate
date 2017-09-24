setwd("C:\\Users\\T430\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data")
# library(memisc)
library(plyr)
library(reshape2)
library(ggplot2)
library(igraph)
library(lme4)
library(MASS)
library(lubridate)
# library(plot3D)
library(colorRamps)

df <- read.table("rec_n500\\dfregall.csv",sep=",",header=T, stringsAsFactors = F)
factorcols <- c('mem_no','pref','pcode','gender','marriage')
# make factors cols into factors
for (col in factorcols) {
  df[,col] <- as.factor(df[,col])
}
# parse date format
df$odate <- ymd_hms(df$odate)


head(df)
dim(df)
length(unique(df$mem_no))

df <- na.omit(df)

## `tenure` removed  -- check if need to add it back in
# rescale
rscols <- c('netWeight','price','point','age','revPC0','revPC1','revPC2','revPC3','qtyPC0','qtyPC1','qtyPC2','qtyPC3')
df[,rscols] <- scale(df[,rscols], scale=T, center=F)


#-------------------------------------------------------------
# CROSS VALIDATION
# MAIN LOOP
# load df; fold; regress; predict; store error; cross-val; repeat
#--------------------------------------------------------------MSE
setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data\\crossval")
rankvec <- c(487, 608, 729, 850, 971)
regvec <- c(1,10,19,28,37,46)
nfolds = 5
MSEarray <- array(NA, c(length(rankvec),length(regvec),nfolds))
for (i in 0:(length(rankvec)-1)){
  for (j in 0:(length(regvec)-1)){
    file <- paste0("dfregall_",i,j,".csv")
    df <- read.table(file, header=T, sep=",", fill = T, stringsAsFactors = F)
    keepcols <- names(df)[which( !(names(df) %in% c('pcode','odate','tenure','revPC2','revPC3','qtyPC2','qtyPC3')) ) ]
    df <- df[,keepcols]

    # data preparation
    df <- na.omit(df)
    # rescale
    rscols <- c('netWeight','price','point','age','revPC0','revPC1',
                'qtyPC0','qtyPC1')
    df[,rscols] <- scale(df[,rscols], scale=T, center=F)

    # make index folds for 10-fold CV
    N <- nrow(df)
    allindices <- seq_len(N)
    index <- list()
    set.seed(111)
    for (k in 1:nfolds){
      leftovers <- allindices[!(allindices %in% unlist(c(index)))]
      index[[k]] <- sample(x=leftovers,size=round(N/nfolds)-1,replace=F)
    }
    # nfolds - cross val
    for (k in 1:nfolds) {
      testind <- index[[k]]
      trainind <- allindices[which(!(allindices %in% testind))]
      dftest <-  df[testind, ]
      dftrain <- df[trainind, ]
      # drop levels from test set that aren't in training set for prediction
#       prefdrop <- levels(dftest$pref)[!(levels(dftest$pref) %in%
#                                           levels(dftrain$pref))]
# #       mem_nodrop <- levels(dftest$mem_no)[!(levels(dftest$mem_no) %in%
# #                                             levels(dftrain$mem_no))]
#       genderdrop <- levels(dftest$gender)[!(levels(dftest$gender) %in%
#                                             levels(dftrain$gender))]
#       marriagedrop <- levels(dftest$marriage)[!(levels(dftest$marriage) %in%
#                                               levels(dftrain$marriage))]
#       dftest <- droplevels(dftest, prefdrop)
#       # dftest <- droplevels(dftest, mem_nodrop)
#       dftest <- droplevels(dftest, genderdrop)
#       dftest <- droplevels(dftest, marriagedrop)

      factorcols <- c('pref','gender','marriage')
      # make factors cols into factors
      for (col in factorcols) {
        dftrain[,col] <- as.factor(dftrain[,col])
        dftest[,col] <- as.factor(dftest[,col])
      }
      # missing levels in test set to NA
      prefid <- which(!(dftest$pref %in% levels(dftrain$pref)))
      genderid <- which(!(dftest$gender %in% levels(dftrain$gender)))
      marriageid <- which(!(dftest$marriage %in% levels(dftrain$marriage)))
      dftest$pref[prefid] <- NA
      dftest$gender[genderid] <- NA
      dftest$marriage[marriageid] <- NA

      # fit model as GLM (no random effects)
      fit1 <- glmer(qty ~ gender + marriage + age + price + point +
                      qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                      netWeight + (1 | pref),
                    family=poisson(link="log"),  data=dftrain,
                    control=glmerControl(optCtrl=list(maxfun=40000),
                                         check.conv.grad=.makeCC("warning",
                                                                 tol = 5e-3,
                                                                 relTol = NULL)
                                         )
                    )
      # predict out-of-sample
      pred1 <- predict(fit1, newdata = dftest, allow.new.levels=T)
      # MSE
      MSEarray[i+1,j+1,k] <- sum( (exp(pred1)-dftest$qty)^2 )/ nrow(dftest)
    } # end nfold cross val

    cat(paste("\ncompleted",i,",",j))
    gc()
  } # end i

}# end j; end MAIN LOOP
dput(MSEarray, "MSEarray_glmer.RData")





# ----------------------------------------------
# plotting error surface
# SELECTION
#---------------------------------------------------

Z <- Zse <- matrix(rep(NA,length(rankvec)*length(regvec)),
                   nrow=length(rankvec))
for (i in 1:length(rankvec)) {
  for (j in 1:length(regvec)){
    # mean
    Z[i,j] <- mean(MSEarray[i,j, ])
    # std error
    Zse[i,j] <- sd(MSEarray[i,j, ]) / sqrt(nfolds)
  }
}


rownames(Z) <- rankvec
colnames(Z) <- regvec
minZ <- min(Z)
minrank <- rankvec[ nfolds - which(min(Z)==Z)%%nfolds ]
minreg <- regvec[ceiling(which(min(Z)==Z) / nrow(Z)) ]
maintitle <- as.character(paste0("MSE Contour\nMin MSE = ",
                            round(minZ,3),", rank = ",
                            round(minrank,1), ", reg. = ",
                            minreg )
                            )

png("MSE_Contour_10v3.png", height=4, width=6, units='in', res=200)
  par(mar=c(4.1,4.1,5,2))
  par(mfrow=c(1,1))
  filled.contour(x=rankvec,y=regvec[1:5],z=Z[1:5,1:5], xlab='rank', ylab='regularization', col=matlab.like(17) , main=maintitle)
  #mtext(maintitle, side=3)
dev.off()












#-----------------------------------------------------
#
# EVALUATION SET CrossValidation
#
#-----------------------------------------------------
setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data\\optimal")
nfolds = 10

file <- "dfregall_opt.csv"
df <- read.table(file, header=T, sep=",", fill = T, stringsAsFactors = F)
keepcols <- names(df)[which( !(names(df) %in% c('pcode','odate','tenure','revPC2','revPC3','qtyPC2','qtyPC3')) ) ]
df <- df[,keepcols]

# data preparation
df <- na.omit(df)
# rescale
rscols <- c('netWeight','price','point','age','revPC0','revPC1',
            'qtyPC0','qtyPC1')
df[,rscols] <- scale(df[,rscols], scale=T, center=F)

# drop missing marraige
droprows <- which(df$marriage == "")
df <- df[-droprows,]
df <- droplevels(df)

# make index folds for 10-fold CV
N <- nrow(df)
allindices <- seq_len(N)
index <- list()
MSE4df <- data.frame(non=rep(NA,nfolds),relational=rep(NA,nfolds))
set.seed(111)
for (k in 1:nfolds){
  leftovers <- allindices[!(allindices %in% unlist(c(index)))]
  index[[k]] <- sample(x=leftovers,size=round(N/nfolds)-1,replace=F)
}
# nfolds - cross val
for (k in 1:nfolds) {
  testind <- index[[k]]
  trainind <- allindices[which(!(allindices %in% testind))]
  dftest <-  df[testind, ]
  dftrain <- df[trainind, ]

  factorcols <- c('pref','gender','marriage')
  # make factors cols into factors
  for (col in factorcols) {
    dftrain[,col] <- as.factor(dftrain[,col])
    dftest[,col] <- as.factor(dftest[,col])
  }
  # missing levels in test set to NA
  prefid <- which(!(dftest$pref %in% levels(dftrain$pref)))
  genderid <- which(!(dftest$gender %in% levels(dftrain$gender)))
  marriageid <- which(!(dftest$marriage %in% levels(dftrain$marriage)))
  dftest$pref[prefid] <- NA
  dftest$gender[genderid] <- NA
  dftest$marriage[marriageid] <- NA

  # fit model as GLM (no random effects)
  fit1 <- glmer(qty ~ gender + marriage + age + price + point +
                  qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                  (1 | pref),
                family=poisson(link="log"),  data=dftrain,
                control=glmerControl(optCtrl=list(maxfun=50000),
                                     check.conv.grad=.makeCC("warning",
                                                             tol = 2e-2,
                                                             relTol = NULL)
                )
  )
  # with RELATIONAL
  fit2 <- glmer(qty ~ gender + marriage + age + price + point +
                  qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                  netWeight + (1 | pref),
                family=poisson(link="log"),  data=dftrain,
                control=glmerControl(optCtrl=list(maxfun=50000),
                                     check.conv.grad=.makeCC("warning",
                                                             tol = 2e-2,
                                                             relTol = NULL)
                )
  )
  # predict out-of-sample
  pred1 <- predict(fit1, newdata = dftest, allow.new.levels=T)
  pred2 <- predict(fit2, newdata = dftest, allow.new.levels=T)
  # MSE
  MSE4df$non[k] <- sum( (exp(pred1)-dftest$qty)^2 )/ nrow(dftest)
  MSE4df$relational[k] <- sum( (exp(pred2)-dftest$qty)^2 )/ nrow(dftest)

  cat(paste0('\nfinished fold ', k))
} # end cross val loop


senon <- sd(MSE3df$non) / sqrt(nrow(MSE4df))
serel <- sd(MSE3df$relational) / sqrt(nrow(MSE4df))
bardf <- data.frame(non=c(mean(MSE4df$non),
                          mean(MSE4df$non)+senon,
                          mean(MSE4df$non)-senon),
                    relational=c(mean(MSE4df$relational),
                    mean(MSE4df$relational)+serel,
                    mean(MSE4df$relational)-serel)
                    )
rownames(bardf) <- c('mu','UCI',"LCI")

write.table(MSE4df,"MSE4df.csv",sep=",")
write.table(bardf,"bardf.csv",sep=",")

print('Finished all processes.')





#-------------------------------------------------
#
## evaluation of entire new sample
#
#-------------------------------------------------
setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data\\optimal")

file <- "dfregall_opt_bothW.csv"
df <- read.table(file, header=T, sep=",", fill = T, stringsAsFactors = F)
keepcols <- names(df)[which( !(names(df) %in% c('pcode','odate','tenure','revPC2','revPC3','qtyPC2','qtyPC3')) ) ]
df <- df[,keepcols]

# data preparation
df <- na.omit(df)
# rescale
rscols <- c('netWeightQty','netWeightRev','price','point','age','revPC0','revPC1',
            'qtyPC0','qtyPC1')
df[,rscols] <- scale(df[,rscols], scale=T, center=F)

factorcols <- c('pref','gender','marriage')
# make factors cols into factors
for (col in factorcols) {
  df[,col] <- as.factor(df[,col])
}

# drop missing marraige
droprows <- which(df$marriage == "")
df <- df[-droprows,]
df <- droplevels(df)

# add moderating variables
df$netWeightQtyMOD <- as.factor(ifelse(df$netWeightQty >=
                                         quantile(df$netWeightQty,.5),1,0) )
df$netWeightRevMOD <- as.factor( ifelse(df$netWeightRev >=
                                          quantile(df$netWeightRev,.5),1,0) )

# WITHOUT Relational
fit1 <- glmer(qty ~ gender + marriage + age + price + point +
                qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                (1 | pref),
              family=poisson(link="log"),  data=df,
              control=glmerControl(optCtrl=list(maxfun=100000),
                                   check.conv.grad=.makeCC("warning",
                                                           tol = 5e-3,
                                                           relTol = NULL)
              )
)

# with RELATIONAL
fit2 <- glmer(qty ~ gender + marriage + age + I(age^2) + price + point +
                qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                netWeightQty + netWeightRev +
                netWeightQty * price +
                netWeightRev * point +
                (1 | pref),
              family=poisson(link="log"),  data=df,
              control=glmerControl(optCtrl=list(maxfun=100000),
                                   check.conv.grad=.makeCC("warning",
                                                           tol = 5e-3,
                                                           relTol = NULL)
              )
)

fit3 <- glmer(qty ~ gender + marriage + age + I(age^2) + price + point +
                qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                netWeightQty + netWeightRev +
                netWeightQty:price +
                netWeightRev:point +
                netWeightRev:netWeightQty +
                netWeightRevMOD:point + netWeightRevMOD:price +
                netWeightQtyMOD:price + netWeightQtyMOD:point +
                netWeightRevMOD:age + netWeightQtyMOD:age+

                (1 | pref),
              family=poisson(link="log"),  data=df,
              control=glmerControl(optCtrl=list(maxfun=100000),
                                   check.conv.grad=.makeCC("warning",
                                                           tol = 5e-3,
                                                           relTol = NULL)
              )
)

summary(fit1)
summary(fit2)
summary(fit3)
lmerTest::anova(fit2,fit3)

save(list = list(fit1=fit1,fit2=fit2), "fitlist_opt_bothW.rda")



dflist <- load("fitlist2_opt.RData")

res1 <- summary(fit1)$res
res2 <- summary(fit2)$res
mse1 <- sum(res1^2) / length(res1)
mse2 <- sum(res2^2) / length(res2)
se1 <- sd(res1) / sqrt(length(res1))
se2 <- sd(res2) / sqrt(length(res2))

msebardf <- data.frame(mse=c(mse1,mse2), se=c(se1,se2))


pred1 <- predict(fit1, newdata = df, allow.new.levels=T)
pred2 <- predict(fit2, newdata = df, allow.new.levels=T)

MSEopt1 <- sum( (exp(pred1)-df$qty)^2 )/ nrow(df)
MSEopt2 <- sum( (exp(pred2)-df$qty)^2 )/ nrow(df)

# ----------------------------------------------
# plotting error surface
# EVALUATION
#---------------------------------------------------

Z <- Zse <- matrix(rep(NA,length(rankvec)*length(regvec)),
                   nrow=length(rankvec))
for (i in 1:length(rankvec)) {
  for (j in 1:length(regvec)){
    # mean
    Z[i,j] <- mean(MSEarray[i,j, ])
    # std error
    Zse[i,j] <- sd(MSEarray[i,j, ]) / sqrt(nfolds)
  }
}


rownames(Z) <- rankvec
colnames(Z) <- regvec
minZ <- min(Z)
minrank <- rankvec[ nfolds - which(min(Z)==Z)%%nfolds ]
minreg <- regvec[ceiling(which(min(Z)==Z) / nrow(Z)) ]
maintitle <- as.character(paste0("MSE Contour\nMin MSE = ",
                                 round(minZ,3),", rank = ",
                                 round(minrank,1), ", reg. = ",
                                 minreg )
)

png("MSE_Contour_10v3.png", height=4, width=6, units='in', res=200)
par(mar=c(4.1,4.1,5,2))
par(mfrow=c(1,1))
filled.contour(x=rankvec,y=regvec[1:5],z=Z[1:5,1:5], xlab='rank', ylab='regularization', col=matlab.like(13) , main=maintitle)
#mtext(maintitle, side=3)
dev.off()




fit1 <- glm(qty ~ gender + marriage + age + price + point +
              qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
              revPC0 + revPC1 + revPC2 + revPC3 +
              netWeight ,
            family=poisson(link="log"),  data=dftrain)
# predict out-of-sample
pred1 <- predict(fit1, newdata = dftest)













#-----------------------------------------------------------
# GLMM models
#-------------------------------------------------------------
glm1a <- glm(qty ~ gender + marriage + age +
              price + point,
            family=poisson(link='log'),
            data=df)
summary(glm1a)

glm1b <- glm(qty ~ gender + marriage + age +
              price + point +
            netWeight,
            family=poisson(link='log'),
            data=df)
summary(glm1b)


glm2a <- glm(qty ~ gender + marriage + age +
              price + point +
              qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
              revPC0 + revPC1 + revPC2 + revPC3 ,
            family=poisson(link='log'),
            data=df)
summary(glm2a)

glm3a <- glm(qty ~ gender + marriage + age +
               price + point +
               qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
               revPC0 + revPC1 + revPC2 + revPC3 +
               netWeight,
             family=poisson(link='log'),
             data=df)
summary(glm3a)

anova(glm2a, glm3a)





me1b <- glmer(qty ~ price + gender + marriage + age+
               (1 | pref),
             family=poisson(link="log"),
             data=df)
summary(me1b)




me2a <- glmer(qty ~ gender + marriage +
                age + price + point +
               qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
               revPC0 + revPC1 + revPC2 + revPC3 +
               (1  | pref),
             family=poisson(link="log"),
             data=df)
summary(me2a)



me3a <- glmer(qty ~ gender + marriage +
                age + price + point +
                qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
                revPC0 + revPC1 + revPC2 + revPC3 +
                netWeight +
                (1  | pref) + (1 | mem_no) ,
              family=poisson(link="log"),
              data=df)
summary(me3a)


# nested
me3a <- glmer(qty ~ gender + marriage +
                age + price + point +
                qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
                revPC0 + revPC1 + revPC2 + revPC3 +
                netWeight +
                (1  | mem_no/pref) ,
              family=poisson(link="log"),
              data=df)
summary(me3a)







anova(me2a,me3a)



dput("cconma_netweight_poisson_mixeff.RData")






n <- 5
prefs <- unique(df$pref)
modlist <- list()
zvec <- rep(NA,length(prefs))
for ( i in 1:length(prefs)) {
  pref_i <- prefs[i]
  data <- df[which(df$pref==pref_i & df$qty>=0),]
  if ( dim(data)[1]>=n ) {
       holder <- glm(qty ~ gender + marriage + age + price+
                       qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
                       revPC0 + revPC1 + revPC2 + revPC3 +
                        netWeight,
                     family=poisson(link='log'),
                      data = data)
         modlist[[length(modlist)+1]] <- holder
         zvec[i] <- summary(holder)$coef[10,3]
#     #ERROR HANDLING
#     possibleError <- tryCatch(
#       summary(modlist[[i]])$coef['z value']['netWeight'],
#       error=function(e) e
#     )
#     if(inherits(possibleError, "error")) next
#     #REAL WORK
#     tvec[i] <- summary(modlist[[i]])$coef['z value']['netWeight']
#     error=function(e) next
#   } else {
#     #modlist[[i]] <- NA
#   }
  }

}


#-------------------------------------------------
# Check significant of netWeight on individual product category regressions
z <- na.omit(unlist(zvec))
q95 <- qnorm(.975, 0, 1)
q99 <- qnorm(.995, 0, 1)

prop95 <- length(z[abs(z)>q95]) / length(z)
prop99 <- length(z[abs(z)>q99]) / length(z)

print(paste0(length(z)," products with >= ", n, " observations"))
print(paste0(round(prop95,3)*100, "% significant at alpha=0.05"))
print(paste0(round(prop99,3)*100, "% significant at alpha=0.01"))

plot(density(z))




length(modlist)



