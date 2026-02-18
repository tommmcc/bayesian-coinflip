# Bayesian Coin Flip

A small probability project focused on explaining Bayesian reasoning in a clear and simple way.

This project was built as an exercise in breaking down a concept that often feels abstract into something easy to follow.

## The Idea

The goal is to estimate the probability of heads, p.

Rather than assigning a single value, p is treated as uncertain and modeled with a probability distribution.

Prior:
p ~ Beta(alpha, beta)

After observing h heads and t tails:

Posterior:
p | data ~ Beta(alpha + h, beta + t)

Because the Beta distribution is conjugate to the Bernoulli likelihood, the update is exact.

## What the Script Does

- Simulates real coin flips and prints the H and T sequence  
- Updates the prior belief using observed data  
- Reports the posterior mean and MAP  
- Calculates a 95 percent Bayesian credible interval  
- Computes probabilities such as P(p > 0.5)

This demonstrates how evidence updates belief and how uncertainty changes as more data is observed.

## Why This Example

Bayesian inference can seem complicated when introduced through heavy notation or large models.

The coin flip example reduces the idea to its essentials:

- A simple likelihood  
- A conjugate prior  
- An exact posterior update  
- Fully interpretable results  

The focus is clarity rather than complexity.

## Run

py bayesiancoinflip.py

Optional reproducible run:

py bayesiancoinflip.py --seed 0

## Example Output

Below is an example run using --seed 3: 

<img width="472" height="357" alt="image" src="https://github.com/user-attachments/assets/7758af70-2295-4268-9806-b0899fb379f3" />






