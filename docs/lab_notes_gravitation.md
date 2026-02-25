# Lab Notes: Newton's Law of Gravitation Discovery

## Experiment Setup
- **Target Law:** $F = G \frac{m_1 m_2}{r^2}$
- **Parameters:**
    - $m_1, m_2 \in [1, 100]$ (Mass)
    - $r \in [1, 10]$ (Distance)
    - $G = 1.0$ (Simplified for structural discovery)
- **Data:**
    - 100 data points generated with the target law.
    - 1% Gaussian noise added to $F$.
- **Dimensional Units:**
    - $m_1, m_2$: [1, 0, 0, 0, 0] (Mass)
    - $r$: [0, 1, 0, 0, 0] (Length)
    - $F$: [1, 1, -2, 0, 0] (Force)

## Hypothesis
Using a "Power Law" primitive set (mul, protected_div) and NSGA-II, the engine should be able to recover the inverse-square relationship and the product of masses.

## Results
- **Success:** Yes. The engine discovered the target structure.
- **Discovered Formula:** $F = 0.9984 \frac{m_1 m_2}{r^2}$
- **MSE:** $8.4981 \cdot 10^7$ (Noisy data)
- **Complexity:** 8.0 (Correct structure)
- **Constants found:** $p_0 = 0.9984$ (Target $G=1.0$)

## Discovered Individual (String)
`0.998386948031628*m1*m2/r**2`

## Discovered Individual (LaTeX)
$$F = 0.9984 \cdot \frac{m_1 \cdot m_2}{r^{2}}$$

## Conclusion
The Newton's Law of Gravitation experiment successfully demonstrates the engine's capability to discover multi-variable power laws using dimensional analysis and hybrid optimization. All steps according to the protocol were completed successfully.
