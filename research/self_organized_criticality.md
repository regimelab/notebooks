SOC‑market(σ̄, ρ, H, τ, η, d, i) ≡
  {
    σ̄ ≪ history     &  // low realized vol “calm”
    ρ_1 ≳ history   &  // autocorr critical slowing
    H > 0.5         &  // long memory / persistence
    τ heavy-tailed  &  // power‑law micro‑moves
    η high          &  // tail risk / kurtotic returns
    d shallow       &  // thin order book
    i increasing    // high impact per unit volume
  } → “asymmetrically sensitive” state:
      same‑sized shock → disproportionately large volatility move
