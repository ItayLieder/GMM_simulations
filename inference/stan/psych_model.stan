data {
    int<lower=1> N;
    vector[N] df;
    int<lower=0,upper=1> y[N];
}

parameters {
    real alpha;
    real<lower=0> sigma;
}
model {

    alpha ~ normal(0,1);
    sigma ~ normal(0,1);

    for (i in 1:N)
        {
            y[i] ~ bernoulli( Phi( df[i] / sigma - alpha ) );
        }

}