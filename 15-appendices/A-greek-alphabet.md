# Appendix A: Greek Alphabet & Pronunciation Guide

> Every Greek letter used in mathematics and ML, with pronunciation, LaTeX commands, and the most common meanings you will encounter in this book and in research papers.

---

## Lowercase & Uppercase

| Lower | Upper | Name | Pronunciation | LaTeX | Common ML/Math Usage |
|-------|-------|------|---------------|-------|---------------------|
| $\alpha$ | $A$ | alpha | AL-fuh | `\alpha` | Learning rate, significance level, angles |
| $\beta$ | $B$ | beta | BAY-tuh | `\beta` | Momentum coefficient, Type II error rate, regression coefficients |
| $\gamma$ | $\Gamma$ | gamma | GAM-uh | `\gamma`, `\Gamma` | Discount factor (RL), Euler-Mascheroni constant, Gamma function |
| $\delta$ | $\Delta$ | delta | DEL-tuh | `\delta`, `\Delta` | Small change, Kronecker delta, $\Delta$ = finite difference |
| $\epsilon$ | $E$ | epsilon | EP-sih-lon | `\epsilon` | Small positive number, machine epsilon, $\epsilon$-greedy exploration |
| $\zeta$ | $Z$ | zeta | ZAY-tuh | `\zeta` | Riemann zeta function (rare in ML) |
| $\eta$ | $H$ | eta | AY-tuh | `\eta` | Learning rate (alternative to $\alpha$), noise |
| $\theta$ | $\Theta$ | theta | THAY-tuh | `\theta`, `\Theta` | Model parameters, angles, $\Theta$ = Big-Theta notation |
| $\iota$ | $I$ | iota | eye-OH-tuh | `\iota` | Rarely used in ML |
| $\kappa$ | $K$ | kappa | KAP-uh | `\kappa` | Condition number, curvature |
| $\lambda$ | $\Lambda$ | lambda | LAM-duh | `\lambda`, `\Lambda` | Regularization strength, eigenvalues, Poisson rate |
| $\mu$ | $M$ | mu | MYOO | `\mu` | Mean, learning rate (physics), Lagrange multiplier |
| $\nu$ | $N$ | nu | NYOO | `\nu` | Degrees of freedom, frequency |
| $\xi$ | $\Xi$ | xi | KSEE / ZAI | `\xi`, `\Xi` | Random variable, slack variable (SVM) |
| $o$ | $O$ | omicron | OH-mih-kron | `o` | Rarely used (conflicts with zero) |
| $\pi$ | $\Pi$ | pi | PIE | `\pi`, `\Pi` | 3.14159..., policy (RL), $\Pi$ = product |
| $\rho$ | $P$ | rho | ROH | `\rho` | Correlation coefficient, density, spectral radius |
| $\sigma$ | $\Sigma$ | sigma | SIG-muh | `\sigma`, `\Sigma` | Standard deviation, sigmoid, $\Sigma$ = summation, covariance matrix |
| $\tau$ | $T$ | tau | TAU (rhymes with cow) | `\tau` | Time constant, temperature (softmax), Kendall's tau |
| $\upsilon$ | $\Upsilon$ | upsilon | OOP-sih-lon | `\upsilon` | Rarely used in ML |
| $\phi$ | $\Phi$ | phi | FEE / FYE | `\phi`, `\Phi` | Feature map, activation function, $\Phi$ = CDF of standard normal |
| $\chi$ | $X$ | chi | KAI | `\chi` | Chi-squared distribution |
| $\psi$ | $\Psi$ | psi | SIGH / PSEE | `\psi` | Wave function (physics), auxiliary function |
| $\omega$ | $\Omega$ | omega | oh-MAY-guh | `\omega`, `\Omega` | Angular frequency, sample space ($\Omega$), weight (alternative) |

---

## The Letters You Will See Most

In roughly descending order of frequency in ML papers:

1. **$\theta$** -- model parameters ("learn $\theta$ to minimize loss")
2. **$\alpha$, $\eta$** -- learning rate
3. **$\lambda$** -- regularization strength, eigenvalues
4. **$\sigma$** -- standard deviation, sigmoid function
5. **$\mu$** -- mean
6. **$\epsilon$** -- small number, noise term
7. **$\beta$** -- momentum, regression coefficients
8. **$\pi$** -- policy (RL), the constant 3.14159...
9. **$\phi$** -- feature map, basis function
10. **$\nabla$** -- gradient (technically not Greek, but lives here: `\nabla`)

---

## Easily Confused Pairs

| Letter | Looks Like | How to Tell Them Apart |
|--------|-----------|----------------------|
| $\nu$ (nu) | $v$ (vee) | $\nu$ has a curly left leg |
| $\rho$ (rho) | $p$ (pee) | $\rho$ has a rounder bowl |
| $\omega$ (omega) | $w$ (double-u) | $\omega$ is rounder, like a curvy "w" |
| $\phi$ (phi) | $\emptyset$ (empty set) | $\emptyset$ has a diagonal slash |
| $\epsilon$ (epsilon) | $\in$ (element of) | $\in$ is a set membership operator |
| $\eta$ (eta) | $n$ (en) | $\eta$ has a descending right stroke |

---

*Back to [Appendices](README.md)*
