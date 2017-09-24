setwd("C:\\Users\\T430\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data")
# library(memisc)
library(texreg)
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


#-----------------------------------------------------
#
# EVALUATION SET CrossValidation
#
#-----------------------------------------------------
nfolds = 10

keepcols <- names(df)[which( !(names(df) %in% c('pcode','odate','revPC2','revPC3','qtyPC2','qtyPC3')) ) ]
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




# add moderating variables
df$netWeightMOD <- as.factor(ifelse(df$netWeight >=
                                         quantile(df$netWeight,.5),1,0) )
# df$netWeightRevMOD <- as.factor( ifelse(df$netWeightRev >=
#                                           quantile(df$netWeightRev,.5),1,0) )


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
); summary(fit1)

# with RELATIONAL
fit2 <- glmer(qty ~ gender + marriage + age + I(age^2) + price + point +
                qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                netWeight + 
                netWeight:price +
                (1 | pref),
              family=poisson(link="log"),  data=df,
              control=glmerControl(optCtrl=list(maxfun=100000),
                                   check.conv.grad=.makeCC("warning",
                                                           tol = 5e-3,
                                                           relTol = NULL)
              )
); summary(fit2)

fit3 <- glmer(qty ~ gender + marriage + age + I(age^2) + price + point +
                qtyPC0 + qtyPC1 + revPC0 + revPC1 +
                netWeight + 
                netWeight:price +  netWeight:point +
                netWeightQtyMOD:price + netWeightQtyMOD:point +
                (1 | pref),
              family=poisson(link="log"),  data=df,
              control=glmerControl(optCtrl=list(maxfun=100000),
                                   check.conv.grad=.makeCC("warning",
                                                           tol = 5e-3,
                                                           relTol = NULL)
              )
); summary(fit3)

screenreg(list(fit1,fit2,fit3), digits = 3)

# summary(fit1)
# summary(fit2)
# summary(fit3)
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



