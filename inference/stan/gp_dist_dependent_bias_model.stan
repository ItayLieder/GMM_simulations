data {
    int<lower=1> N; // number of points
    vector[N] df;  // f1(t) - f2(t)
    vector[N] d1;  // f1(t) - (f1+f2)(t-1)
    vector[N] d2;  // f1(t) - (f1+f2)(t-2)
    real<lower=0> beta3; // gp white noise
    real<lower=0> beta1; // gp scale
    real<lower=0> beta2; // gp inverse squared distance scale
    int<lower=0,upper=1> y[N]; // binary response (f1 higher)
}

transformed data {
    vector[N] mu; // Prior mean of the Gaussian process
    mu <- rep_vector(0, N);
}

parameters {
    vector[N] alpha1; // intercept as a function of d1
    vector[N] alpha2; // intercept as a function of d2
    real<lower=0> sigma; // std sensory noise in Probit
}
model {

    matrix[N, N] Sigma1;
    matrix[N, N] Sigma2;

    for(i in 1:N)
    for(j in i:N){
        Sigma1[i, j] <- beta1 * exp(- pow(d1[i] - d1[j], 2) / beta2)
            + if_else(i==j, beta3, 0.0);
        Sigma2[i, j] <- beta1 * exp(- pow(d2[i] - d2[j], 2) / beta2)
            + if_else(i==j, beta3, 0.0);
    }
    for(i in 1:(N-1))
        for(j in (i+1):N){
            Sigma1[j, i] <- Sigma1[i, j];
            Sigma2[j, i] <- Sigma2[i, j];
    }

    alpha1 ~ multi_normal(mu, Sigma1);
    alpha2 ~ multi_normal(mu, Sigma2);
    sigma ~ normal(0,0.5);
    for (i in 1:N)
        {
            y[i] ~ bernoulli( Phi( df[i] / sigma - alpha1[i] - alpha2[i] ) );
        }

}