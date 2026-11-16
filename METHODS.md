# Method

The harness answers one question. How many runs does a pass rate gate need before a drop in agent success is distinguishable from ordinary run to run variation? `harness/runcount.py` answers it with exact arithmetic, so the same command prints the same text on every run. `harness/gates_sim.py` then checks those counts on a mock agent.

## The two gates

Both gates use a significance level of 5% (alpha 0.05).

The one sample gate applies when the baseline pass rate is already known. It runs the candidate n times and counts the successes. The cutoff c is the largest success count whose cumulative binomial probability, computed at the baseline rate, is still at or below alpha. The build fails when the observed successes are at or below c. That is the largest cutoff that holds the false alarm rate at or below alpha when nothing has changed.

The two sample gate applies when the baseline is measured too. Both the baseline and the candidate get n runs. The build fails when a one sided Fisher exact test says the candidate pass rate is lower, at the same significance level. For each possible total number of successes across the two arms, the cutoff is the largest candidate success count whose one sided Fisher probability is at or below alpha.

## How the run count search works

The one sample search tries n from 5 up to 1,500. It reports two counts. The first is the smallest n whose power reaches the target. Binomial power is not monotone in n, so it also reports the smallest n from which the power stays at or above the target for every larger n up to 1,500.

The two sample search starts from a normal approximation, then checks exact power in a window around it. The window runs from 20 below the approximation (never below 5) to 400 above it. An earlier version of the script stopped 40 runs above the approximation, and the exact count for the drop from 95% to 92% sits 55 runs above that approximation, so the script crashed. The window above the approximation is now 400 runs. The reported count is the first n inside that window whose exact power reaches the target.

## Judge noise

A judge that sometimes flips a correct verdict changes the pass rate the gate observes. With a symmetric flip chance e, an agent whose true pass rate is p is seen at p times (1 minus e) plus (1 minus p) times e. The same two sample search then runs on those observed rates. The gate is unchanged. Only the rates it is given are different.

## The simulation

`harness/gates_sim.py` uses a mock agent. Each run succeeds with a fixed probability, 95% when the agent is healthy and 85% when it has regressed, and a smaller drop to 90% is checked as well. A correct run matches one golden answer with probability 0.40, which stands in for an answer that is right but worded differently.

The simulation runs 10,000 builds with seed 20261029. It compares three gates on the same mock agent: an exact match against the golden answer, a single run outcome check, and the two sample pass rate gate. The pass rate gate is spot checked at the two run counts sized for 80% power, 126 runs per side for the drop from 95% to 85% and 371 runs per side for the drop from 95% to 90%. The script also prints the closed form probability next to each simulated rate.

## Versions

The commands were run on Python 3.9.6 with numpy 2.0.2. The same outputs were also checked on numpy 2.2.6.

## Limits

The agent is a mock, so these results describe the gates and not a language model. Runs are treated as independent, and every scenario is treated as equally hard. The 10,000 builds use one seed.
