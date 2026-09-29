# Brokerage Pricing Model Redesign

## Case Question

Suppose an online financial brokerage wants to redesign its pricing model. How
should the analysis determine the best pricing structure while protecting
long-term customer value, profitability, customer experience, fairness, and
regulatory compliance?

This document is an analytical framework, not evidence that the underlying
models or experiments have already been implemented.

## Recommended Analytical Approach

```text
Business objective and guardrails
        ↓
Pricing and unit-economics inventory
        ↓
Point-in-time analytical dataset
        ↓
Customer segmentation
        ↓
Causal price-response identification
        ↓
Scenario simulation with uncertainty
        ↓
Constrained pricing optimization
        ↓
Controlled experiment
        ↓
Staged rollout and monitoring
```

### 1. Define the decision

Agree with finance, product, risk, compliance, and customer teams on the
primary objective: contribution margin, long-term customer value, retention,
net deposits, trading activity, or another measurable outcome. Define
guardrails before considering candidate prices: churn, complaints, support
demand, withdrawals, excessive trading, fairness, transparency, and regulatory
risk.

### 2. Build the economic and causal data foundation

Create a customer-period dataset containing point-in-time prices, promotions,
trades, AUM, deposits, margin usage, revenue, execution costs, FX costs,
servicing costs, tenure, acquisition channel, market conditions, churn,
complaints, and support contacts. Record eligibility, price exposure time,
and outcome windows. Exclude post-treatment variables from causal features.

### 3. Establish identification before modelling

Define the treatment, unit, treatment date, exposure window, outcome window,
control group, and causal estimand. Test whether historical price changes,
promotions, geographic variation, product variation, or customer-tier
variation provide credible identification.

Check for confounding, reverse causality, selection bias, positivity
violations, interference between customers, market-regime changes, seasonality,
survivorship bias, and policy changes. Prefer randomized experiments. Where
appropriate, consider difference-in-differences, fixed effects, matching, or
heterogeneous treatment-effect models with explicit assumptions.

### 4. Segment customers and estimate heterogeneous response

Evaluate high-frequency traders, casual investors, high-AUM customers,
savings-plan customers, margin users, new customers, and mature customers.
Estimate how price changes affect trading, churn, deposits, AUM, complaints,
and contribution margin for each eligible segment. Distinguish:

```text
Predictive: Who is likely to trade?
Causal: How would behaviour change if the price changed?
```

Predictive accuracy is not evidence of causal price sensitivity.

### 5. Simulate pricing scenarios

Compare flat fees, volume tiers, subscriptions, hybrid pricing, product-level
pricing, caps, minimums, and discounts. Include commissions, spreads, FX,
margin interest, execution, servicing, incentives, compliance, and support
costs. Treat the simulator as scenario analysis unless its response functions
are causally calibrated.

Report point estimates, confidence intervals, elasticity sensitivity, downside
scenarios, and market-regime stress cases. Classify each input as:

```text
Observed | Estimated | Causally identified | Assumed | Scenario-only | Unknown
```

### 6. Optimize under constraints

Optimize contribution margin or long-term customer value only subject to
customer, fairness, regulatory, churn, complaint, and customer-experience
constraints. Individualized prices, sensitive or proxy-sensitive features,
large fee increases, margin changes, and changes affecting vulnerable groups
require human and compliance approval.

### 7. Experiment before launch

Specify the eligible population, treatment, control, randomization unit, sample
size, duration, primary metric, secondary metrics, guardrails, stopping rules,
exclusions, compliance approval, and rollback plan. Do not launch a pricing
change unless the causal evidence, uncertainty, segment stability, fairness,
compliance, explainability, rollback, and monitoring gates pass.

### 8. Monitor after launch

Monitor contribution margin, revenue per customer, trades, retention, AUM,
deposits, complaints, support contacts, price-elasticity drift, segment drift,
market-regime changes, and customer-harm indicators. Define alert thresholds,
re-estimation triggers, and rollback rules before launch.

## Required Decision Output

For every pricing recommendation, answer:

1. What can be identified from the available data?
2. What cannot be identified?
3. What assumptions are required?
4. What data is missing?
5. Is causal price elasticity identifiable?
6. Is the simulator calibrated or scenario-only?
7. What experiment is required?
8. What customer-harm guardrails are required?
9. What fairness, regulatory, and consumer-protection constraints apply?
10. What evidence is required before production launch?

If credible price variation is unavailable, state explicitly:

```text
Causal elasticity cannot currently be identified.
```
