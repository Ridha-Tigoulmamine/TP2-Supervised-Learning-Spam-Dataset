#TP2-classification supervisé (Régression linéaire-KNN-bayésien naïf)
#Ridha TIGOULMAMINE
#Houcine FORLOUL
#Sara BAICHE
#Rachda BOUGDOUR  
#Année universitaire : 2020 - 2021


library(kernlab)
data(spam)
attach(spam)
#Description
summary(spam)
?spam
str(spam)
names(spam)
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
#analyse statistique univariée : : l'étude d'une seule variable*********************************************************
summary(type)
sd(spam$address)
sd(spam$your)
par(mfrow=c(2,2))
summary(free)
hist(free,col =1,xlim=c(0,2),breaks=100)
summary(money)
hist(money,col =2,xlim=c(0,1),breaks=50)
summary(num000)
hist(num000,col =3,xlim=c(0,1),breaks=20)
summary(charDollar)
hist(charDollar,col =4,xlim=c(0,1),breaks=50)
head(type)
tail(type)
#analyse statistique bivariée : l'étude des relations entre deux variables***********************************************
library(lares)
corr_cross(spam,top=5)                #Meilleures variables corrélées entre eux  
corr_var(spam,type,top =5)            #Meilleures Variables corrélées avec "type" 
library(lattice)
splom(~spam[,c('your','credit','money')], groups=g)
splom(~spam[,21:25], groups=g)
pairs(spam[,21:23], col=as.numeric(g))
#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------regression lineare----------------------------------------------------------
#**********************************************************************************************************************
#-------------------------------------------sans normalisation----------------------------------------------------------err=  0.1119322
library(kernlab)
data(spam)
attach(spam)
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
y <- ifelse(g=="spam", 1, 0) 
x=spam[,1:57]
lm.fit <- lm(y~.,spam[,1:57])    
lm.beta <- lm.fit$coef 
yhat <- predict(lm.fit) 
lm.ghat <- factor(ifelse(yhat > 0.5, "1", "0")) 
sum(lm.ghat != y)                   # nombre d'exemples mal class??s 
mean(lm.ghat != y)                  # erreur de classification 
table(lm.ghat, y)                   # matrice de confusion 
#------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------regression lineare---------------------------------------------------------
#*************************************************************************************************************************
# ---avec normalisation diviser chaque case nij de la table spam par la racine carrée du produit des sommes marginale ni. Et n.j--------------------------------
#------------------------------------------------------------------------------------------------------------------------err=0.1388829
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
y <- ifelse(g=="spam", 1, 0) 
X1=spam[,1:57]
spam_normalize<-matrix( ncol=57, nrow=4601, byrow=FALSE)
nj=colSums(X1)                      #la somme marginale nj
ni=rowSums(X1)                      #la somme marginale ni
for (i in 1:nrow(X1))               #divisé chaque case x[i,j] sur la racine carré de produit des somme ni[i] et nj[j]
{ 
  for (j in 1:ncol(X1)) 
  {
    spam_normalize[i,j] = (X1[i,j]/sqrt(ni[i]*nj[j]))
  }
}
set.seed(30)
X=cbind(y,spam_normalize)
lm.fit <- lm(y~spam_normalize) 
lm.beta <- lm.fit$coef 
yhat <- predict(lm.fit) 
lm.ghat <- factor(ifelse(yhat > 0.5, "1", "0")) 
sum(lm.ghat != y)                   # nombre d'exemples mal class??s 
mean(lm.ghat !=y)                  # erreur de classification 
table(lm.ghat, y)                   # matrice de confusion 
#------------------------------------------------------------------------------------------------------------------------
#----------------------------------knn avec 75% train  25 % test----------------------------------------------------------------
#*********************************************************************************************************************
#----------------------------------sans normalisation----------------------------------------------------------------- k=12   err_test=0.1998262
#------------------------------------------------------------------------------------------------------------------------
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
library(class)
set.seed(30)
X=cbind(g,spam[,1:57])
tr <- sample(1:nrow(X),3450)
Xtrain <- X[tr,]
Xtest <- X[-tr,]
kmax=50
err_test <- rep(NA,kmax)
for (k in 1:kmax)
{
  pred <- knn(Xtrain[,-1],Xtest[,-1],Xtrain[,1],k)
  err_test[k] <- sum(pred!=Xtest[,1])/length(Xtest[,1])
}
lim <- c(0,max(err_test))
plot(err_test,type="l",ylim=lim,col=2,xlab="nombre de voisins",
     ylab= "taux d'erreur")
which.min(err_test)
min(err_test) 
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------knn avec cross validation-----------------------------------------------------
#***********************************************************************************************************************
#------------------------------------------------sans normalisation----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------k=3 err=0.1798436
library(class)
kmax=100
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=cbind(g,spam[,1:57])
?knn.cv
err_valid <- rep(NA,kmax)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
for (k in 1:kmax) 
{ 
  pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
  err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
} 
which.min(err_valid) 
pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
sum(pred!=Xtest[,1])/length(Xtest[,1])
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------KNN avec 75% train 25% test--------------------------------------------------------------
#**************************************************************************************************************************************************************************
#-----------------normalisation diviser chaque case nij de la table spam par la racine carrée du produit des sommes marginale ni. Et n.j----------------
#------------------------------------------------------------------------------------------------------------------------   // k=8 err=0.08601216
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,1:57]
spam_normalize<-matrix( ncol=57, nrow=4601, byrow=FALSE)
nj=colSums(X1)
ni=rowSums(X1)
for (i in 1:nrow(X1)) 
{ 
  for (j in 1:ncol(X1)) 
  {
    spam_normalize[i,j] = (X1[i,j]/sqrt(ni[i]*nj[j]))
  }
}
library(class)
set.seed(30)
X=cbind(g,spam_normalize)
tr <- sample(1:nrow(X),3450)
Xtrain <- X[tr,]
Xtest <- X[-tr,]
kmax=50
err_test <- rep(NA,kmax)
for (k in 1:kmax)
{
  pred <- knn(Xtrain[,-1],Xtest[,-1],Xtrain[,1],k)
  err_test[k] <- sum(pred!=Xtest[,1])/length(Xtest[,1])
}
#which.min(err_test) #k=1
which.min(err_test[2:50])
#min(err_test)  #erreur de k=1
min(err_test[2:50]) 
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------KNN avec cross validation----------------------------------------------------------
#***********************************************************************************************************************
#---------------normalisation diviser chaque case nij de la table spam par la racine carrée du produit des sommes marginale ni. Et n.j---------
#------------------------------------------------------------------------------------------------------------------------k=2 err= 0.105126
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,1:57]
spam_normalize<-matrix( ncol=57, nrow=4601, byrow=FALSE)
nj=colSums(X1)
ni=rowSums(X1)
for (i in 1:nrow(X1)) 
{ 
  for (j in 1:ncol(X1)) 
  {
    spam_normalize[i,j] = (X1[i,j]/sqrt(ni[i]*nj[j]))
  }
}
set.seed(30)
X=cbind(g,spam_normalize)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
library(class)
kmax=50
err_valid <- rep(NA,kmax)
for (k in 1:kmax) 
{ 
  pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
  err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
} 
k1=which.min(err_valid[2:50])
k1
pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k1) 
sum(pred!=Xtest[,1])/length(Xtest[,1])
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------bayésien naïf-----------------------------------------------------------
#*************************************************************************************************************************
#----------------------------------------------sans normalisation-------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------err=0.2864595
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=spam[,1:57]
library(e1071)
spam.d<- X # ensemble apprentissage 
m <- naiveBayes(g ~ ., data = spam.d) 
## alternativement: 
m <- naiveBayes(spam.d, g) 
#m 
a=table(predict(m, spam.d), g) 
a
err_nai=(a[1,2]+a[2,1])/nrow(spam.d)
err_nai
#------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------bayésien naïf---------------------------------------------------------
#*************************************************************************************************************
#---------------------------------------------avec normalisation------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ err=0.3040643
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,1:57]
spam_normalize<-matrix( ncol=57, nrow=4601, byrow=FALSE)
nj=colSums(X1)
ni=rowSums(X1)
for (i in 1:nrow(X1)) 
{ 
  for (j in 1:ncol(X1)) 
  {
    spam_normalize[i,j] = (X1[i,j]/sqrt(ni[i]*nj[j]))
  }
}
X=cbind(g,spam_normalize)
library(e1071)
spam.d<- X # ensemble apprentissage 
m <- naiveBayes(g ~ ., data = spam.d) 
## alternativement: 
m <- naiveBayes(spam.d, g) 
#m 
a=table(predict(m, spam.d), g) 
a
err_nai=(a[1,2]+a[2,1])/nrow(spam.d)
err_nai
#------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------bayésien naïf-------------------------------------------------------------
#***********************************************tout les variables************************************************
#--------------------------------------------*avec normalisation scale()----------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ err =0.2599435
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=spam[,1:57]
X1=scale(X)
X=cbind(g,X1)
library(e1071)
spam.d<- X # ensemble apprentissage 
m <- naiveBayes(g ~ ., data = spam.d) 
## alternativement: 
m <- naiveBayes(spam.d, g) 
#m 
a=table(predict(m, spam.d), g) 
a
err_nai=(a[1,2]+a[2,1])/nrow(spam.d)
err_nai
#--------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Dans la partie qui suit nous avons fait plusiers d'autres test :                                    !
#utiliser les méthodes de classification avec un choix des varibles                                  !
#utiliser la fonction scale() pour normaliser les donnéer                                            !
#utiliser des déffirentes découpages pour trouvé K pour KNN                                          !
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




#-------------------------------------------regression lineare-------------------------------------------------------------------
#*******************************************tous les variables*******************************************************************
#-------------------------------------------avec normalisation (scale) -----------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------err =0.1119322
X=spam[,1:57]
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
y <- ifelse(g=="spam", 1, 0) 
X_normalization=scale(X)
lm.fit <- lm(y~X_normalization)
lm.beta <- lm.fit$coef 
yhat <- predict(lm.fit) 
lm.ghat <- factor(ifelse(yhat > 0.5, "1", "0")) 
sum(lm.ghat != y)                   # nombre d'exemples mal class??s 
mean(lm.ghat != y)                  # erreur de classification 
table(lm.ghat, y)                   # matrice de confusion 
#------------------------------------------------------------------------------------------------------------------------
#------------------------------KNN avec plusieurs découpages des données en deux parties "apprentissage-validation" et "test----------------------
#******************************tous les variables**********************************"***************************************
#------------------------------sans normalisation------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=1 err=0,186087
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=cbind(g,spam[,1:57])
B <- 10 
kmax <- 50 
err_valid <- rep(NA,kmax) 
err_test <- rep(NA,B) 
for (b in 1:B) 
{ 
  tr <- sample(1:nrow(X),3450)
  Xtrainval <- X[tr,]
  Xtest <- X[-tr,]
  for (k in 1:kmax) 
  {
    pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
    err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
  } 
  pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid[2:50]+1)) 
  err_test[b] <- sum(pred!=Xtest[,1])/length(Xtest[,1]) 
} 
par(mfrow=c(1,1))
boxplot(err_test,main="Erreurs test pour 50 decoupages")
which.min(err_valid)        
min(err_valid)
#------------------------------------------------------------------------------------------------------------------------
#-----------------KNN avec plusieurs découpages des données en deux parties "apprentissage-validation" et "test--------------------------------------
#*************************************************tous les variables****************************************
#-----normalisation diviser chaque case nij de la table spam par la racine carrée du produit des sommes marginale ni. Et n.j--------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=3 err=0.09101449
X1=spam[,1:57]
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
spam_normalize<-matrix( ncol=57, nrow=4601, byrow=FALSE)
nj=colSums(X1)
ni=rowSums(X1)
for (i in 1:nrow(X1)) 
{ 
  for (j in 1:ncol(X1)) 
  {
    spam_normalize[i,j] = (X1[i,j]/sqrt(ni[i]*nj[j]))
  }
}
library(class)
set.seed(30)
X=cbind(g,spam_normalize)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
B <- 10 
kmax <- 50 
err_valid <- rep(NA,kmax) 
err_test <- rep(NA,B) 
for (b in 1:B) 
{ 
  for (k in 1:kmax) 
  {
    pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
    err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
  } 
  pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
  err_test[b] <- sum(pred!=Xtest[,1])/length(Xtest[,1]) 
} 
a=which.min(err_valid[2:50])+1
a
min(err_valid[2:50])
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------regression lineare----------------------------------------------------------
#*******************************************avec choix des variables**********************************************************
#----------------------------------------------sans normalisation-------------------------------------------------------------err=0.2147359
y <- ifelse(g=="spam", 1, 0) 
lm.fit <- lm(y~your+num000+remove)    #les variables choisi par rapport la correlation avec variables type pour la suite de l'etudes your+num000+remove  
lm.beta <- lm.fit$coef 
b <- -lm.beta[2]/lm.beta[4] 
a <- (0.5 - lm.beta[1] - lm.beta[3]*mean(num000))/lm.beta[4] 
par(mfrow=c(1,1))
plot(your, remove, col=g) 
abline(a,b) 
yhat <- predict(lm.fit) 
lm.ghat <- factor(ifelse(yhat > 0.5, "1", "0")) 
sum(lm.ghat != y)                   # nombre d'exemples mal class??s 
mean(lm.ghat != y)                  # erreur de classification 
table(lm.ghat, y)                   # matrice de confusion 
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------regression lineare-------------------------------------------------------------
#************************************avec choix des variables************************************************************
#------------------------------------avec normalisation méthode (Scale) -----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------err=0.2147359
library(MASS)
library(cluster)
library(clusterSim)
X=spam[,c('your','num000','remove')]
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
y <- ifelse(g=="spam", 1, 0) 
#X_normalization=data.Normalization (X,type="n1",normalization="column")
X_normalization=scale(X)
lm.fit <- lm(y~X_normalization[,c('your')]+X_normalization[,c('num000')]+X_normalization[,c('remove')]) 
lm.beta <- lm.fit$coef 
b <- -lm.beta[2]/lm.beta[4] 
a <- (0.5 - lm.beta[1] - lm.beta[3]*mean(X_normalization[,c('num000')]))/lm.beta[4] 
yhat <- predict(lm.fit) 
lm.ghat <- factor(ifelse(yhat > 0.5, "1", "0")) 
sum(lm.ghat != y)                   # nombre d'exemples mal class??s 
mean(lm.ghat != y)                  # erreur de classification 
table(lm.ghat, y)                   # matrice de confusion 
#------------------------------------------------------------------------------------------------------------------------
#----------------------------------------knn avec 75% train  25 % test--------------------------------------------------------
#********************************************avec choix des variables**********************************************************
#-----------------------------------------sans normalisation----------------------------------------------------------------- 
#--------------------------------------------------------------------------------------------------------------------------------- k=9   err_test=0.1468288
?knn
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
library(class)
set.seed(30)
X=cbind(g,spam[,c('your','num000','remove','charDollar','you')])
tr <- sample(1:nrow(X),3450)
Xtrain <- X[tr,]
Xtest <- X[-tr,]
kmax=50
err_test <- rep(NA,kmax)
for (k in 1:kmax)
{
  pred <- knn(Xtrain[,-1],Xtest[,-1],Xtrain[,1],k)
  err_test[k] <- sum(pred!=Xtest[,1])/length(Xtest[,1])
}
lim <- c(0,max(err_test))
plot(err_test,type="l",ylim=lim,col=2,xlab="nombre de voisins",
     ylab= "taux d'erreur")
which.min(err_test)
min(err_test) 
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------knn avec cross validation---------------------------------------------------
#*************************************************avec choix des variables********************************************
#----------------------------------------------------sans normalisation-------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=5 err=0.1668115
library(class)
kmax=100
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=cbind(g,spam[,c('your','num000','remove','charDollar','you')])
?knn.cv
err_valid <- rep(NA,kmax)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
for (k in 1:kmax) 
{ 
  pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
  err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
} 
which.min(err_valid) 
pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
sum(pred!=Xtest[,1])/length(Xtest[,1])
#------------------------------------------------------------------------------------------------------------------------
#------------------------------KNN avec plusieurs découpages des données en deux parties "apprentissage-validation" et "test------------------------------------------------
#*************************************************avec choix des variables********************************************
#----------------------------------------------------sans normalisation-------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=6
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=cbind(g,spam[,c('your','num000','remove','charDollar','you')])
B <- 10 
kmax <- 50 
err_valid <- rep(NA,kmax) 
err_test <- rep(NA,B) 
for (b in 1:B) 
{ 
  tr <- sample(1:nrow(X),3450)
  Xtrainval <- X[tr,]
  Xtest <- X[-tr,]
  for (k in 1:kmax) 
  {
    pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
    err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
  } 
  pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
  err_test[b] <- sum(pred!=Xtest[,1])/length(Xtest[,1]) 
} 
boxplot(err_test,main="Erreurs test pour 50 decoupages")
which.min(err_valid)        
min(err_valid)
#------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------KNN avec 75% train 25% test ------------------------------------------------
#*******************************************avec choix des variables **************************************
#------------------------------------------avec normalisation avec commande scale-----------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=10 err_test = 0.1190269
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,c('your','num000','remove','charDollar','you')]
X_normalization=scale(X1)
library(class)
set.seed(30)
X=cbind(g,X_normalization)
tr <- sample(1:nrow(X),3450)
Xtrain <- X[tr,]
Xtest <- X[-tr,]
kmax=50
err_test <- rep(NA,kmax)
for (k in 1:kmax)
{
  pred <- knn(Xtrain[,-1],Xtest[,-1],Xtrain[,1],k)
  err_test[k] <- sum(pred!=Xtest[,1])/length(Xtest[,1])
}
lim <- c(0,max(err_test))
plot(err_test,type="l",ylim=lim,col=2,xlab="nombre de voisins",
     ylab= "taux d'erreur")
which.min(err_test)
min(err_test)
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------KNN avec cross validation--------------------------------------------------------
#***************************************avec choix des variables**************************************
#--------------------------------------avec normalisation scale()---------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=12 err_test = 0.1303215
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,c('your','num000','remove','charDollar','you')]
X_normalization=scale(X1)
library(class)
set.seed(30)
X=cbind(g,X_normalization)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
kmax=100
err_valid <- rep(NA,kmax)
for (k in 1:kmax) 
{ 
  pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
  err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
} 
which.min(err_valid) 
pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
sum(pred!=Xtest[,1])/length(Xtest[,1])
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------KNN avec plusieurs découpages des données en deux parties "apprentissage-validation" et "test --------------------
#*********************************************avec choix des variables***************************************
#-------------------------------------------avec normalisation Scale() --------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------k=16 err=0.1423188
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X1=spam[,c('your','num000','remove','charDollar','you')]
X_normalization=scale(X1)
library(class)
set.seed(30)
X=cbind(g,X_normalization)
tr <- sample(1:nrow(X),3450)
Xtrainval <- X[tr,]
Xtest <- X[-tr,]
B <- 10 
kmax <- 50 
err_valid <- rep(NA,kmax) 
err_test <- rep(NA,B) 
for (b in 1:B) 
{ 
  for (k in 1:kmax) 
  {
    pred <- knn.cv(Xtrainval[,-1],Xtrainval[,1],k) 
    err_valid[k] <- sum(pred!=Xtrainval[,1])/length(Xtrainval[,1]) 
  } 
  pred <- knn(Xtrainval[,-1],Xtest[,-1],Xtrainval[,1],k=which.min(err_valid)) 
  err_test[b] <- sum(pred!=Xtest[,1])/length(Xtest[,1]) 
} 
boxplot(err_test,main="Erreurs test pour 50 decoupages")
which.min(err_valid)        
min(err_valid)
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------bayésien naïf----------------------------------------------------------
#********************************************avec le choix des variables**********************************************************
#---------------------------------------------sans normalisation---------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------err=0.04064334
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=cbind(g,spam[,c('your','num000','remove','charDollar','you')])
library(e1071)
spam.d<- X # ensemble apprentissage 
m <- naiveBayes(g ~ ., data = spam.d) 
## alternativement: 
m <- naiveBayes(spam.d, g) 
#m 
a=table(predict(m, spam.d), g) 
a
err_nai=(a[1,2]+a[2,1])/nrow(spam.d)
err_nai
#------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------bayésien naïf---------------------------------------------------------
#*******************************************avec choix des variables************************************************
#-------------------------------------------avec normalisation scale()-------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ err=0.009345794
g <- factor(ifelse(type != "spam", "nonspam", "spam")) 
X=spam[,c('your','num000','remove','charDollar','you')]
X1=scale(X)
X=cbind(g,X1)
library(e1071)
spam.d<- X # ensemble apprentissage 
m <- naiveBayes(g ~ ., data = spam.d) 
## alternativement: 
m <- naiveBayes(spam.d, g) 
#m 
a=table(predict(m, spam.d), g) 
a
err_nai=(a[1,2]+a[2,1])/nrow(spam.d)
err_nai
