# min |x|_1 subject to Ax = b
# library(CVXR)
l1eq_pd <- function(x0, A, b){
  n <- length(x0)
  if (is.complex(x0) | is.complex(A) | is.complex(b)){
    x <- Variable(n, complex=TRUE)
  }else{
    x <- Variable(n)
  }
  objective <- Minimize(CVXR::norm1(x))
  constraints <- list(A %*% x == b)
  problem <- Problem(objective, constraints)
  result <- solve(problem, solver='SCS')
  return(result$getValue(x))
}