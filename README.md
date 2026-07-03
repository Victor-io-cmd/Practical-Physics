# gum-calc

A calculation engine that turns raw lab measurements into a fully formatted uncertainty analysis, ready to paste into a LaTeX report.

---

## What this is

In experimental physics, every measurement comes with uncertainty, and every lab report needs a rigorous section explaining where that uncertainty comes from and how it propagates through the final result. This process follows an international standard called the **GUM** (Guide to the Expression of Uncertainty in Measurement), and doing it by hand for every quantity in a report is slow, repetitive, and error-prone: one skipped derivative or one misapplied rounding rule can silently produce a wrong result.

`gum-calc` automates that entire pipeline. Give it a formula, some measured values, and how each value was obtained (repeated measurements, instrument resolution, manufacturer tolerance...), and it returns the propagated result together with a ready-to-use LaTeX writeup: the measurement model, the sensitivity coefficients, the uncertainty budget, and the final result correctly rounded.

I built this for my own physics lab reports as an L3 Physics student at UPEC (Université Paris-Est Créteil), and I use it every time I need to write an uncertainty section.

---

## Author

**Victorio BONNEVILLE DIAZ** — L3 Physics student, UPEC (Université Paris-Est Créteil)

Code generated with [Claude (Anthropic)](https://www.anthropic.com) from an architecture and logic defined by the author, based on the GUM metrology coursework from L2 Physics at Université Paris-Est Créteil.

---

## Project structure

```
gum-calc/
├── GUM - Measurement Uncertainties.ipynb   # Notebook template
├── gum_calc.py                             # Calculation engine and LaTeX export
└── README.md
```

---

## Technology

- **Python** for the calculation engine
- **SymPy** for symbolic differentiation — the engine computes exact partial derivatives of any formula instead of relying on hardcoded propagation formulas
- **SciPy** for the Student's t-distribution table, used in the Welch-Satterthwaite widening factor
- **LaTeX / siunitx** as the output format, so the result can be pasted directly into a report compiled in Overleaf
- **Jupyter Notebook** as the working environment, one cell per measured quantity

---

## Features

**Uncertainty typing.** Five functions cover the standard GUM cases: repeated measurements (type A), instrument resolution or known tolerance (type B), a directly known standard uncertainty (for example, propagated from a previous calculation), and exact constants.

**Symbolic propagation.** The core function takes a formula as plain text (like `"U / I"`), parses it with SymPy, and differentiates it automatically with respect to every variable. This is what removes the need to derive propagation formulas by hand for every new measurand.

**Welch-Satterthwaite widening factor.** When uncertainty sources of different reliability are combined (a well-known type B source alongside a type A source estimated from only a few measurements), the engine computes the effective degrees of freedom and reads the correct widening factor `k` from the Student's t table, instead of always assuming `k = 2`.

**Linear regression with covariance.** Ordinary least squares fitting that also returns the covariance between the slope and intercept, a quantity that is almost always forgotten in hand calculations but that matters whenever a final result depends on both.

**Correct rounding, every time.** The uncertainty always decides the number of significant digits shown for the result, never the other way around — one of the most common mistakes in lab reports. This rule lives in a single function so it can never be applied inconsistently across a report.

**LaTeX generation.** Full write-ups using the `siunitx` package: the measurement model, a description of each uncertainty source, the sensitivity coefficients, the Welch-Satterthwaite step when relevant, the uncertainty budget, and the boxed final result. Multiple write-ups can be assembled into a single appendix section.

**Fails loudly, not silently.** Nearly every function checks its own inputs before calculating anything, and raises a precise error describing exactly what is wrong, rather than letting a bad value slip through and produce a wrong number in a report.

---

## The process

The project started from a real, recurring problem: rewriting the same uncertainty calculations and the same LaTeX blocks for every TP, with the same risk of making a rounding or propagation mistake each time.

I designed the architecture and the calculation logic myself, based on the GUM metrology course from L2 Physics at UPEC: which uncertainty types exist, how propagation and widening are supposed to work, and what a correct uncertainty budget looks like. From there, I worked with Claude (Anthropic) to translate that logic into working Python — writing the SymPy-based propagation engine, the LaTeX formatting layer, and the regression pipeline, then testing edge cases against what the GUM actually prescribes.

The codebase is deliberately split into two layers that never mix: a calculation layer that knows nothing about LaTeX, and a formatting layer that runs no physics, only reusing results the calculation layer already produced. That separation means the output format can change without ever touching a formula, and a formula can change without ever touching a rounding rule.

---

## What I learned

**On the physics side**, working through every uncertainty type forced me to actually understand *why* the GUM formulas look the way they do, not just apply them. Writing the Welch-Satterthwaite step, for instance, made the distinction between a type A and a type B source concrete: one is a statistical estimate with limited degrees of freedom, the other is (in general) a fixed, well-known bound. Implementing the linear regression covariance term was the clearest example: it's the kind of correlation that's easy to miss by hand, and building it into the engine means it's now impossible to forget.

**On the engineering side**, this was my first real experience directing an AI collaborator on a project I designed myself rather than asking it to solve a problem from scratch. The useful split turned out to be: I own the domain logic and the architecture decisions (what a function should take as input, what "correct" means for a given GUM rule), and the AI translates that into code, catches edge cases I hadn't thought of, and keeps the implementation consistent across a large file. That only works if the underlying logic is mine — an AI can write correct code around a wrong idea just as easily as around a right one, so understanding the GUM well enough to check the output mattered more than the code itself.

---

## How can it be improved

A few limitations are already documented directly in the code, and are the natural next steps:

- **Covariance is opt-in, not automatic.** Right now, the engine only accounts for correlation between two input quantities if it's explicitly passed in. The one common case that's handled automatically is a measurand built from a regression's slope and intercept. Any other source of correlation (say, two different measurands derived from the same regression) currently has to be handled manually by the caller.
- **Propagation is strictly linear (first order).** The engine uses partial derivatives evaluated at the nominal value, which is what the GUM prescribes for well-behaved formulas. For strongly non-linear models, the GUM's own supplement recommends a Monte Carlo approach instead — that's not implemented yet.
- **No weighted regression.** The linear fit currently assumes every data point carries the same uncertainty on y. Real experimental data doesn't always work that way, and a weighted least squares option would make the regression more broadly usable.
- **Unit algebra is symbolic, not numeric.** When combining units (for example, deriving a slope's unit from `y_unit / x_unit`), the engine treats unit names as tokens rather than physical quantities. It won't automatically simplify `\kilo\gram` against `\gram` — mixing prefixes for the same physical unit currently produces an unsimplified (but not incorrect) result.

Beyond these, a natural extension would be a small test suite comparing the engine's output against hand-verified GUM examples from the course, to catch regressions automatically as the code evolves.