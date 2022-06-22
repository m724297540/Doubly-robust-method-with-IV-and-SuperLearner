
fold <- function(Y, X, T, Z, k){
  data <- do.call("cbind", list(Y, X, T, Z))
  loc <- sample(seq(1, nrow(data)))
  fold <- cut(loc, breaks = k , labels = FALSE)
  return(fold)
}

# nuisance function
driv.superlearner.nuisance <- function(Y,X,T,Z,k = 5,...){
  #create folds
  fold <- fold(Y=Y, X=X, T=T, Z=Z, k=k)
  
  # data input
  data <- do.call("cbind", list(Y,X,T,Z))
 
  # set up variables for later use
  n.sample <- nrow(data)
  pre.y <- c(rep(NA, n.sample))
  pre.t <- c(rep(NA, n.sample))
  pre.z <- c(rep(NA, n.sample))
  pre.tz <- c(rep(NA, n.sample))
  pre.theta <- c(rep(NA, n.sample))
  res.y <- c(rep(NA, n.sample))
  res.t <- c(rep(NA, n.sample))
  res.z <- c(rep(NA, n.sample))
  res.tz <- c(rep(NA, n.sample)) 
  TZ <- T*Z
  
  # set up learner library
  SL.ranger.new = function(...){
    SL.ranger(..., num.trees = 1000)
  }
    learners <- c("SL.ranger.new", "SL.xgboost", "SL.nnet", "SL.glmnet")
  
  # CV control for the superlearner
  control <- SuperLearner.CV.control(V = 10)
  

  for(i in 1:k){
    
      # set up test and training sample
      test.ind <- which(fold == i, arr.ind = TRUE)
      X.train <- X[-test.ind,]
      X.test <- X[test.ind,]
      
      # predict y on x to get pre.y
      # calculate res.y = y - pre.y
      y.mod <- SuperLearner(Y = Y[-test.ind], X = X.train, newX = X.test,
                            SL.library = learners, verbose = FALSE, 
                            method = "method.NNLS", cvControl = control)
      pre.y[test.ind] <- y.mod$SL.predict
      res.y[test.ind] <- Y[test.ind] - pre.y[test.ind]
      
      # predict t on x to get pre.t
      # calculate res.t = t - pre.t
      t.mod <- SuperLearner(Y = T[-test.ind], X = X.train, newX = X.test,
                            SL.library = learners, verbose = FALSE, 
                            method = "method.NNLS", cvControl = control)
      pre.t[test.ind] <- t.mod$SL.predict
      res.t[test.ind] <- T[test.ind] - pre.t[test.ind]
      
      # predict z on x to get pre.z
      # calculate res.z = z - pre.z
      z.mod <- SuperLearner(Y = Z[-test.ind], X = X.train, newX = X.test,
                            SL.library = learners, verbose = FALSE, 
                            method = "method.NNLS", cvControl = control)
      pre.z[test.ind] <- z.mod$SL.predict
      res.z[test.ind] <- Z[test.ind] - pre.z[test.ind]
      
      # predict tz on x to get pre.tz
      # calculate res.tz = pre.tz - pre.t * pre.z
      tz.mod <- SuperLearner(Y = TZ[-test.ind], X = X.train, newX = X.test,
                            SL.library = learners, verbose = FALSE, 
                            method = "method.NNLS", cvControl = control)
      pre.tz[test.ind] <- tz.mod$SL.predict
      res.tz[test.ind] <- TZ[test.ind] - pre.tz[test.ind]
      
      # predict pre.theta
      lm <- dmliv.superlearner(Y[-test.ind], X[-test.ind,], T[-test.ind],
                      Z[-test.ind], k = k)
      pre.theta[test.ind] <- predict(lm, X[test.ind,])
  }
  
    result <- list("res.y" = res.y, "res.t" = res.t, "res.z" = res.z,
                 "res.tz" = res.tz, "pre.theta" = pre.theta)
}   

# Double Machine Learning CATE with IV
dmliv.superlearner <- function(Y,X,T,Z,k = 5, ...){
  #create folds
  fold <- fold(Y=Y, X=X, T=T, Z=Z, k=k)
  
  #data input
  data <- do.call("cbind", list(Y,X,T,Z))
  n.sample <- nrow(data)
  
  #set up variables for later use
  pre.y <- c(rep(NA, n.sample))
  pre.t.zx <- c(rep(NA, n.sample))
  pre.t.x <- c(rep(NA, n.sample))
  zx <- as.data.frame(cbind(Z,X))
  
  # learner library
  SL.ranger.new = function(...){
    SL.ranger(..., num.trees = 1000)
  }
    learners <- c("SL.ranger.new", "SL.xgboost", "SL.nnet", "SL.glmnet")
    
  # CV control for the superlearner
  control <- SuperLearner.CV.control(V = 10)
  
  for(i in 1:k){
    test.ind <- which(fold == i, arr.ind = TRUE)
    X.train <- X[-test.ind,]
    X.test <- X[test.ind,]
    
    # predict y on x to get pre.y
    # calculate res.y = y - pre.y
    y.mod <- SuperLearner(Y = Y[-test.ind], X = X.train, newX = X.test,
                          SL.library = learners, verbose = FALSE, 
                          method = "method.NNLS", cvControl = control)
    pre.y[test.ind] <- y.mod$SL.predict
    
    # predict t on x to get pre.t
    # calculate res.t = t - pre.t
    
    t.x.mod <- SuperLearner(Y = T[-test.ind], X = X.train, newX = X.test,
                          SL.library = learners, verbose = FALSE, 
                          method = "method.NNLS", cvControl = control)
    pre.t.x[test.ind] <- t.x.mod$SL.predict
    
    # predict t on z, x to get pre.t.zx
    # calculate res.z = z - pre.z
    
    t.zx.mod <- SuperLearner(Y = T[-test.ind], X = zx[-test.ind,], 
                             newX = zx[test.ind,],
                             SL.library = learners, verbose = FALSE, 
                             method = "method.NNLS", cvControl = control)
    pre.t.zx[test.ind] <- t.zx.mod$SL.predict
  }
  
  # using lm for final stage
  res.y <- Y - pre.y
  t <-pre.t.zx - pre.t.x
  t.sign <- sign(t)
  t.sign[t.sign == 0] <- 1
  clipped.t <- t.sign * ramify::clip(abs(t), 1e-6, Inf)
  weight <- clipped.t * clipped.t
  y <- res.y / clipped.t
  
  traindata <- as.data.frame(cbind(y, X))
  lm <- lm(y~., data = traindata, weights = weight)
  return(lm)
}

# Double Machine Learning ATE estimation with IV
dmlateiv.superlearner <- function(Y,X,T,Z,k=5,alpha = 0.05,...){
  #create folds
  fold <- fold(Y=Y, X=X, T=T, Z=Z, k=k)
  
  #data input
  data <- do.call("cbind", list(Y,X,T,Z))
  n.sample <- nrow(data)
  
  #set up variables for later use
  pre.y <- c(rep(NA, n.sample))
  pre.t <- c(rep(NA, n.sample))
  pre.z <- c(rep(NA, n.sample))
  res.y <- c(rep(NA, n.sample))
  res.t <- c(rep(NA, n.sample))
  res.z <- c(rep(NA, n.sample))
  
  
  # learner library
  SL.ranger.new = function(...){
    SL.ranger(..., num.trees = 1000)
  }
  learners <- c("SL.ranger.new", "SL.xgboost", "SL.nnet", "SL.glmnet")
  
  # CV control for the superlearner
  control <- SuperLearner.CV.control(V = 10)
  
  for(i in 1:k){
    test.ind <- which(fold == i, arr.ind = TRUE)
    X.train <- X[-test.ind,]
    X.test <- X[test.ind,]
    
    # predict y on x to get pre.y
    # calculate res.y = y - pre.y
    y.mod <- SuperLearner(Y = Y[-test.ind], X = X.train, newX = X.test,
                          SL.library = learners, verbose = FALSE, 
                          method = "method.NNLS", cvControl = control)
    pre.y[test.ind] <- y.mod$SL.predict
    res.y[test.ind] <- Y[test.ind] - pre.y[test.ind]
    
    # predict t on x to get pre.t
    # calculate res.t = t - pre.t
    
    t.mod <- SuperLearner(Y = T[-test.ind], X = X.train, newX = X.test,
                          SL.library = learners, verbose = FALSE, 
                          method = "method.NNLS", cvControl = control)
    pre.t[test.ind] <- t.mod$SL.predict
    res.t[test.ind] <- T[test.ind] - pre.t[test.ind]
    
    # predict z on x to get pre.z
    # calculate res.z = z - pre.z
    
    z.mod <- SuperLearner(Y = Z[-test.ind], X = X.train, newX = X.test,
                          SL.library = learners, verbose = FALSE, 
                          method = "method.NNLS", cvControl = control)
    pre.z[test.ind] <- z.mod$SL.predict
    res.z[test.ind] <- Z[test.ind] - pre.z[test.ind]
  }
  
    #calculate ate theta as E[res_y * res_z] /E[res_t * res_z]
    theta <-  mean(res.y * res.z)/mean(res.t * res.z)
    std <- sd(res.y * res.z)/(sqrt(n.sample) * abs(mean(res.t * res.z)))
    lb <- qnorm(p = alpha/2, mean = theta, sd = std)
    ub <- qnorm(p = 1 - alpha/2, mean = theta, sd = std)
    CI <- c(lb, ub)
    result <- list("theta" = theta, "std" = std, "CI" = CI)
    return(result)
}

# Doubly Robust quantity
y_nu <- function(Y,X,T,Z,k=5,tz.clip = NULL){
  
  n <- driv.superlearner.nuisance(Y = Y,X = X,T = T,Z = Z,k = k)
  pre.theta <-n$pre.theta
  res.y <- n$res.y
  res.t <- n$res.t
  res.z <- n$res.z
  res.tz <- n$res.tz 
  
  if (is.null(tz.clip)){
    y.nu <- pre.theta + (res.y - pre.theta * res.t) * res.z / res.tz
  } else {
      tz.sign <- sign(res.tz)
      tz.sign[tz.sign == 0] <- 1
      clipped.tz <- tz.sign * ramify::clip(abs(res.tz), tz.clip, Inf)
      y.nu <- pre.theta + (res.y - pre.theta * res.t) * res.z / clipped.tz
    }
  return(y.nu)
}

# Double Robust wiht IV
driv.final <- function(Y,X,T,Z, k = 5, method,formula = NULL, ...) {
  
  y_nu <- y_nu(Y = Y, X = X, T = T, Z = Z, k = k,...)
  
  if (method == "constant"){
    const <- c(rep(1, dim(X)[1]))
    lm <- lm(y_nu ~ const - 1)
    return(summary(lm))
  } else if(method == "lm"){
    if (is.null(formula)){
      # if function is not specified, use all X
      new_data <- as.data.frame(cbind(y_nu, X))
      lm <- lm(y_nu ~., data = new_data)
      pre <- predict(lm, new_data)
      sum <- summary(lm)
      result <- list("summary" = sum, "cate" = pre, "lm" = lm)
      return(result)
    } else {
        new_data <- as.data.frame(cbind(y_nu, X))
        lm <- lm(formula, data = new_data)
        pre <- predict(lm, new_data)
        sum <- summary(lm)
        result <- list("summary" = sum, "cate" = pre, "lm" = lm)
        return(result)
      } 
  } else if(method == "rf"){
    if (is.null(formula)){
      # if function is not specified, use all X
      new_data <- as.data.frame(cbind(y_nu, X))
      rf <- ranger(y_nu ~., data = new_data)
      pre <- predict(rf, new_data)
      result <- list("cate" = pre$predictions, "forest" = rf)
      return(result)
    } else{
      new_data <- as.data.frame(cbind(y_nu, X))
      rf <- ranger(formula, data = new_data)
      pre <- predict(rf, new_data)
      result <- list("cate" = pre$predictions, "forest" = rf)
      return(result)
    }
  } 
}    
    
