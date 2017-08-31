library(deepnet)
require(nnet)
library(RcppDL)
library(keras)

#data processing
iris <- read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", col_names = FALSE) 
iris[,5] <- as.numeric(as.factor(unlist(iris[,5]))) -1
iris <- as.matrix(iris)
dimnames(iris) <- NULL

#for RBMmoed
iris_x <- normalize(iris[,1:4])
iris_mat <- cbind(iris_x, iris[,5])
head(iris_mat)

#
ind <- sample(2, nrow(iris_x), replace=TRUE, prob=c(0.67, 0.33))
iris.training <- iris_x[ind==1, 1:4]
iris.test <- iris_x[ind==2, 1:4]

#Training a Deep neural network with weights initialized by DBN
#dnn <- dbn.dnn.train(x, y, hidden = c(5, 5))
dnn <- dbn.dnn.train(iris_x, iris[,5], hidden = c(5, 5))

#Training a RBM(restricted Boltzmann Machine)
#rbm <- rbm.train(x3, 10, numepochs = 20, cd = 10)
rbm <- rbm.train(iris_mat, 10, numepochs = 20, cd = 10)

#Training Neural Network
#nn <- nn.train(x, y, hidden = c(5))
nn <- nn.train(iris_x, iris[,5], hidden = c(5))

#Training a Deep neural network with weights initialized by Stacked AutoEncoder
#dnn <- sae.dnn.train(x, y, hidden = c(5, 5))
saednn <- sae.dnn.train(iris_x, iris[,5], hidden = c(5, 5))


#predict
PredictResult <- nn.predict(saednn, iris.test)

PredictResult

