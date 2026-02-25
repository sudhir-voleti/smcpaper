data {
  int<lower=1> N_total;        // Total observations (N_cust * T)
  int<lower=1> N_cust;         // Number of customers
  array[N_cust] int<lower=1> T; // Time points per customer
  int<lower=1> K;              // Number of states (3: Dormant, Lukewarm, Whale)
  vector[N_total] y;           // Observed spend
}

parameters {
  // Transition Matrix
  array[K] simplex[K] Gamma;   
  
  // Emission Parameters (Hurdle-Gamma)
  vector<lower=0, upper=1>[K] pi0; // Probability of zero per state
  vector<lower=0>[K] alpha;        // Gamma shape per state
  vector<lower=0>[K] beta;         // Gamma rate per state
}

transformed parameters {
  // Log-transformations for the forward algorithm
  array[K] vector[K] log_Gamma;
  for (k in 1:K) log_Gamma[k] = log(Gamma[k]);
}

model {
  // --- Priors ---
  // Sticky priors for transitions (diagonal bias)
  for (k in 1:K) {
    vector[K] dir_prior = rep_vector(1.0, K);
    dir_prior[k] = 5.0; // Encourage staying in the same state
    Gamma[k] ~ dirichlet(dir_prior);
  }
  
  // Emission Priors (Informed by your "4 Worlds" logic)
  pi0 ~ beta(1, 1); 
  alpha ~ exponential(0.1); 
  beta ~ exponential(0.1);

  // --- Forward Algorithm (Likelihood) ---
  int pos = 1;
  for (c in 1:N_cust) {
    vector[K] log_p_y; // Log-likelihood of y[pos] for each state
    vector[K] acc;     // Accumulator for forward probabilities
    
    // t = 1 (Assume steady state or uniform initial)
    for (k in 1:K) {
      if (y[pos] == 0) {
        log_p_y[k] = log(pi0[k]);
      } else {
        log_p_y[k] = log1m(pi0[k]) + gamma_lpdf(y[pos] | alpha[k], beta[k]);
      }
    }
    acc = log_p_y - log(K); // Uniform initial state p=1/K
    pos += 1;

    // t = 2 to T
    for (t in 2:T[c]) {
      vector[K] next_acc;
      for (k in 1:K) {
        // Log-likelihood of observation at time t in state k
        real log_p_obs;
        if (y[pos] == 0) {
          log_p_obs = log(pi0[k]);
        } else {
          log_p_obs = log1m(pi0[k]) + gamma_lpdf(y[pos] | alpha[k], beta[k]);
        }
        
        // Marginalize previous state
        next_acc[k] = log_sum_exp(acc + log_Gamma[k]) + log_p_obs;
      }
      acc = next_acc;
      pos += 1;
    }
    target += log_sum_exp(acc);
  }
}
