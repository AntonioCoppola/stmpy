# E-Step for a Document Block, using Spark
beta <- rep(list(beta),1)
beta.index = betaindex
update.mu
lambda.old = lambda


# Initialize a Spark SQL context
sqlContext <- sparkRSQL.init(sc)
df = createDataFrame(sqlContext, faithful)

lines <- textFile(sc, "README.md")
wordsPerLine <- lapply(lines, function(line) { length(unlist(strsplit(line, " "))) })

# Useful constants
V <- ncol(beta[[1]])
K <- nrow(beta[[1]])
N <- length(documents)
A <- length(beta)

if(!update.mu) mu.i <- as.numeric(mu)
  
# 1) Initialize Sufficient Statistics 
sigma.ss <- diag(0, nrow=(K-1))
beta.ss <- vector(mode="list", length=A)
for(i in 1:A) {
  beta.ss[[i]] <- matrix(0, nrow=K,ncol=V)
}
  bound <- vector(length=N)
  lambda <- vector("list", length=N)
  
  # 2) Precalculate common components
  sigobj <- try(chol.default(sigma), silent=TRUE)
  if(class(sigobj)=="try-error") {
    sigmaentropy <- (.5*determinant(sigma, logarithm=TRUE)$modulus[1])
    siginv <- solve(sigma)
  } else {
    sigmaentropy <- sum(log(diag(sigobj)))
    siginv <- chol2inv(sigobj)
  }
  # 3) Document Scheduling
  # For right now we are just doing everything in serial.
  # the challenge with multicore is efficient scheduling while
  # maintaining a small dimension for the sufficient statistics.
  for(i in 1:N) {
    #update components
    doc <- documents[[i]]
    words <- doc[1,]
    aspect <- beta.index[i]
    init <- lambda.old[i,]
    if(update.mu) mu.i <- mu[,i]
    beta.i <- beta[[aspect]][,words,drop=FALSE]
    
    #infer the document
    doc.results <- logisticnormalcpp(eta=init, mu=mu.i, siginv=siginv, beta=beta.i, 
                                  doc=doc, sigmaentropy=sigmaentropy)
    
    # update sufficient statistics 
    sigma.ss <- sigma.ss + doc.results$eta$nu
    beta.ss[[aspect]][,words] <- doc.results$phis + beta.ss[[aspect]][,words]
    bound[i] <- doc.results$bound
    lambda[[i]] <- c(doc.results$eta$lambda)
    if(verbose && i%%ctevery==0) cat(".")
  }
  if(verbose) cat("\n") #add a line break for the next message.
  
  #4) Combine and Return Sufficient Statistics
  lambda <- do.call(rbind, lambda)
  return(list(sigma=sigma.ss, beta=beta.ss, bound=bound, lambda=lambda))
}

diag(0, nrow=(K-1))
