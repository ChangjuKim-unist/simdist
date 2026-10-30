# Stage-1 Review: "Identify, Then Cover: Where Coverage Comes From in Simulation-Pretrained World-Model Planners"

Reviewer role: RSS area chair, adversarial read. Line numbers (Lnn) refer to `main.tex`. The compiled PDF is 4 of the allowed 5 pages and contains no figure.

Reference points used throughout: SimDist (Levy et al. 2026), DayDreamer (Wu 2022), TD-MPC/TD-MPC2 (Hansen 2022/2024), RMA (Kumar 2021), HIM (Long 2024), DrEureka (Ma 2024), DROPO (Tiboni 2023), ASID (Memmel 2024), SPI-Active (Sobanbabu 2025), SimOpt (Chebotar 2019, not cited), BayesSim (Ramos 2019, not cited), Ross & Bagnell 2012, MBPO (Janner 2019), PDML (Wang 2023).

---

## 1. Central claim: is it one sentence, and is it stable?

There are three different central claims, and the paper switches between them.

- **Abstract (L26), finding (i):** "the ability to operate under a new dynamics regime is set *entirely* by the planner's sampling prior." A diagnostic claim about SimDist-style pipelines.
- **Intro (L32):** "Our findings reverse the usual order of effort... coverage is a property of pretraining, not of adaptation." Same claim, now a normative one ("order of effort").
- **Intro (L34) / Sec. IV (L72) / title:** the claim becomes a *recipe*: identify by replay, retrain the prior to cover, plan, adapt only on identified-sim data.
- **Conclusion (L81):** "Real data is therefore best spent on identifying the physics, after which coverage is bought in simulation." A third formulation: a claim about the *best use of real data*.

Drift that a reviewer will flag:

1. "Set entirely by the prior" (L26) is contradicted in the same abstract by (ii) "the planner recovers the prior's compounding error inside its coverage (falls ×0.2–0.5)" and by Table I (B0/D = 0.83 [0.74, 0.93] at w = 0.5). "Entirely" should be "the reachable regime is set by the prior; the fall rate inside it is not."
2. "Coverage" is used in two senses: the DR interval of the expert (L42, L52) and "the ability to operate under a regime" (L26). Table I shows these are not the same thing: w = 0.7 "covers" 0.7 but the prior falls 9.6/1000 at 0.4, and w = 0.5 does not cover 0.4 but falls drop to 4.3. The paper never defines coverage as a measurable quantity.
3. "Identify, then cover" is in the title and Sec. IV, but the only experiment that would test *identification-driven* covering (the targeted w = 0.3–0.7 prior) is marked "results pending" (L72). The title names an untested recipe; what is tested is "blanket-widen, then don't adapt."
4. The thesis on real data shifts between sections. L32: real data is dangerous "in a specific way... if the data were collected by a different prior." L66: "data from the deployed prior never hurt." L72: on-policy data from the deployed prior at the true condition raised falls 1.49 → 2.74 (ratio 1.84 [1.25, 2.51]). So real data collected by the deployed prior *did* hurt, in a way that is not coupling. The Limitations (L78) calls this "an observation, not a general claim," but step (4) of the recipe is built on exactly this observation.

**Verdict:** the claim is clear sentence by sentence but not stable. The honest one-sentence version, consistent with every number in the draft, is: *"In a SimDist-style pipeline with frozen encoder, value and BC prior, real-data finetuning of the latent dynamics cannot move the robot into an actuator regime the prior's randomization did not reach; retraining the prior in simulation to cover an identified regime can."*

---

## 2. Novelty, contribution by contribution

| # | Contribution (L34) | Verdict | Would be cited against it |
|---|---|---|---|
| 1 | Component-wise measurement of what adaptation, planning and the prior can fix | **Incremental.** That DR range bounds what a distilled policy can do is the premise of RMA (fails beyond 8 kg without the adaptation module), Peng et al. 2018, and every ADR paper. That a sampling-based planner cannot escape its policy prior is already shown by the TD-MPC/TD-MPC2 ablations (planning without the policy prior collapses) and by the 5 % prior-sample design of SimDist itself. The *new* part is the specific negative result for SimDist's "finetune f only" recipe under an out-of-DR actuator regime. That is a useful finding, but it is a finding about one pipeline's frozen components, not about world-model planners in general. DayDreamer shows real data *can* establish a new regime when nothing is frozen; the draft never scopes its claim against that. | RMA, TD-MPC2, DayDreamer, Peng 2018 |
| 2 | Coupling of model/prior updates to the collecting policy | **Known.** The draft itself concedes this (L37: "reach the same conclusion"). Ross & Bagnell 2012 give the concentrability bound; MBPO, PDML (Wang 2023) and the offline-RL distribution-shift literature are the empirical side. "We provide the first hardware-relevant measurement" (L37) is unearned: every coupling number in the draft is sim-to-sim. The one thing that is new here is a magnitude (×2.2 / ×2.8) for a latent world model in a locomotion setting; that is a data point, not a result. | Ross & Bagnell 2012, Janner 2019, Wang 2023, Levine 2020 (offline RL) |
| 3 | Policy-independent replay identifier + the identify-then-cover recipe | **Already known as a method.** DROPO (cited, L37) is nearly the same procedure: offline logs, replay in simulation under candidate parameters, trajectory-error likelihood, posterior over DR parameters, *then train the policy in simulation under the inferred distribution*. SimOpt (Chebotar et al. 2019, not cited) is literally "identify from real rollouts, then re-randomize and retrain." SPI-Active (cited) does it for a legged robot. The only candidate-new piece is the contrast "history probe R^2 ≈ 0 vs. replay identifies" (L69). But the project's own notes show why: the torque cap almost never activates in the expert's DR training data, so the regime is invisible *because the training data lacks excitation*, while the replay uses deployment logs at the weak condition where saturation does occur. That is an excitation argument, which is exactly ASID's and SPI-Active's thesis. Stated that way it is a confirmation, not a contribution. | DROPO, SimOpt, BayesSim, SPI-Active, ASID |
| 4 | Hardware protocol with software torque caps as ground-truth actuator weakening | **Modestly new as a benchmark design; no results.** Software torque caps giving a known scale is a good idea and I have not seen it used as a sim2real identification ground truth on the Go1. But it is a protocol (L75), with "Preliminary results will be inserted here" as its result. For Stage 1 this is acceptable only if the protocol is specified to the level of a pre-registration (see §6). | — |

Net: a knowledgeable reviewer will say the paper re-discovers DROPO/SimOpt inside the SimDist pipeline and attaches a well-executed negative result about SimDist's adaptation step. The negative result is the publishable core. The recipe is not.

---

## 3. Strength of evidence

### Supported by numbers with CIs
- Planner helps inside coverage: B0/D = 0.83 [0.74, 0.93] at w = 0.5 (Table I); ratios 0.22–0.51 "all intervals excluding 1" at nominal and 0.6× (L55). Supported.
- Coupling of f: 2.15 [1.64, 2.78] vs. 1.00 [0.84, 1.19]; of π: 2.83 [2.30, 3.57] vs. 1.12 [0.88, 1.43] (Table II). Supported, for the cloned broad prior.
- Nominal improvement from on-policy data: ×0.74 [0.55, 0.98] (L66). Supported (barely).

### Over-stated or unsupported
1. **Three incompatible numbers for "narrow prior at 0.4× torque."** Table I "none" column: 10.2 (prior alone) / 10.3 (with planner). L52: "3.8 to 3.2." L52 again: "SimDist 4.1." These are 2.5× apart for what the text presents as one system. From the project notes, 3.8/3.2/0.9 come from an earlier run with a different broad student and D′ = 4.31 for the narrow student, i.e. a different seed/dataset. The draft never says so. The Limitations (L78) then admits "seed variance is comparable to the width effect for the cloned priors." Put together: **Table I is a one-seed sweep whose between-column differences are of the same size as seed noise, and the text's headline numbers come from a different seed of the same experiment.** A reviewer will treat Table I's "curve" as unestablished.
2. **"Fourfold" (L26).** True for the one-seed cloned sweep (10.3 → 2.2, 4.7×). The fair comparison the draft itself proposes (L42: SimDist's own head vs. the retrained broad head) gives 4.1 → 1.5 = 2.7×. The abstract advertises the larger, less fair number.
3. **"3.2 to 0.9 with no real data" (L52)** vs. Table I's 2.2 for the same configuration (broad prior + planner, pretrained model, 0.4×). Same claim, numbers differ 2.4×, no explanation.
4. **"The planner... does not change this curve" (L52) and "Outside the coverage it changes nothing" (L55).** Table I shows B0/D = 1.13 and 1.14 at w = 0.9, 0.7 *with the CIs omitted*. The project's CSV has [1.09, 1.17] and [1.11, 1.16], both excluding 1. So the planner significantly *worsens* falls outside coverage, and the draft hides this by dropping two CIs from a row that has CIs in its other three cells. This will be read as selective reporting.
5. **Sec. IV, identified-sim adaptation (L72):** "reduced falls at the true 0.4 from 1.49 to 1.24 ([0.62, 1.11])." The bracket is a CI on the *ratio* (0.83), not on 1.24, and it includes 1. The sentence should say "no significant change." Likewise "2.74 ([1.25, 2.51])": 2.74 is not inside its own interval; it is the ratio 1.84 whose CI this is. Two mislabeled intervals in the one paragraph that justifies step (4) of the recipe.
6. **Step (4) of the recipe is confounded by identification error.** In the sim-to-sim benchmark "real logs" at the true condition *are* on-policy simulator data at s = 0.4. "Identified-simulator data" is on-policy simulator data at s = 0.5. The only difference between the harmful and the safe dataset is that the identified scale is biased 25 % upward, which makes the data milder (project notes: 30/30 episodes fall-free at 0.5 vs. 62 episodes averaging 725 steps at 0.4). So the evidence that "identified-sim data is safe" is equivalently evidence that "data from a wrong, easier simulator is safe." The notes also attribute the upward bias partly to the identifier scaling `effort_limit` while the evaluation scales `saturation_effort` too, i.e. the identified simulator and the evaluation simulator differ by construction. None of this is in the draft. Without a control (on-policy sim data at the *true* s = 0.4 adapted on the broad head is the 2.74 row; there is no "identified = true" row), step (4) is unsupported.
7. **"One minute of logs" (L26, L34, L72).** Every simulation experiment used 45k-step (15-minute) logs; the hardware plan adapts SimDist on 29.5 min. The identifier's data requirement was never measured. Unsupported.
8. **"From nominal logs it is 1.03" (L69).** The notes show the likelihood is flat above 0.7 (the cap never engages), with MAP at the grid end (1.20) and posterior mean 1.03. The draft reports MAP for the 0.4 logs and posterior mean for the nominal logs without saying which estimator is used. One estimator, stated, for all three numbers.
9. **"Probe R^2 ≈ 0" → "invisible to the prior" (L26, L69).** Invisible in the expert's DR data, where saturation rarely occurs (adding the true scale to the history does not change next-state predictability in the notes). The draft generalizes a data-distribution property to an axis property; the replay identifier's own weighting ("steps where the logged PD torque demand approaches the limit," L69) shows identifiability depends on saturation events in both cases.
10. **"10 per 1000 steps for every prior whose randomization stops above 0.7×" (L26).** w = 0.7 is included in the set, so "at or above."
11. **"We provide the first hardware-relevant measurement of this coupling" (L37).** No hardware measurement exists in the draft.
12. **"the fall-heavy saturated dynamics is learned at the expense of the frozen value head's calibration" (L72).** Stated as a mechanism; it is a hypothesis (the notes say "가설"). No experiment on V.

---

## 4. Is "identify, then cover" a method?

As written, no. It is four known steps in sequence: DROPO-style replay identification (DROPO/SimOpt), DR retraining on the identified interval (SimOpt/DROPO), BC distillation into the policy head (SimDist), and a *don't-do* (skip real-data adaptation). What would make it a method:

- **Algorithm box** with inputs (logs, grid, segment length, demand weight, likelihood temperature, estimator) and outputs (interval for the expert's DR). None of the identifier's parameters are in the draft: candidate grid (0.2–1.2, 21 points in the code), replicas per candidate (18), number of segments/rounds, segment length (0.4 s now; the design doc's original was 0.2 s), the (max|τ|/τ_nom)^2 weighting, the error→likelihood map, and MAP vs. posterior mean. "Reset to a logged state" also needs: which states (joint q/q̇, base angular/linear velocity, gravity vector → roll/pitch, height from foot kinematics per the code), how the first two steps are excluded for contact transients, and how fallen-robot segments are handled. The identifier is not reproducible from the text.
- **What is automated vs. hand-chosen.** "Randomization that contains the identified scale with margin" (L72): the margin (0.3–0.7 around 0.5) is hand-chosen. A method would derive the interval from the posterior (e.g. the 5–95 % posterior quantiles widened by the identifier's measured bias). The sampled fraction (5 %), MPPI noise and iteration count are also fixed by hand and never varied, yet the claim "the planner cannot extend coverage" depends on them.
- **Ablations missing.** (a) Targeted (0.3–0.7) vs. blanket (0.3–1.1) vs. narrow: this is the P10f experiment, pending; it is the *only* experiment that tests the title. (b) Planner sample fraction / noise scale vs. coverage. (c) Unfreezing V during adaptation (the draft's own hypothesis for the harm). (d) Seeds: ≥3 experts per width. (e) Identifier accuracy vs. log length (to support "one minute").
- **Failure cases.** Friction not identifiable (stated, good). Motor/payload trade-off in the joint grid (stated). Missing: what happens when the identified scale is wrong by the measured bias, i.e., the robot at 0.4 and the prior trained for 0.5–0.7.
- **Cost.** The "cover" step is a new expert (≈2.5 GPU-h) plus a new head (≈2 h) per identified regime, offline between sessions. SimDist's selling point is minutes of real data; the recipe replaces that with hours of retraining and a second session. The draft should say this plainly; a reviewer will otherwise assume it is being hidden.

---

## 5. Writing

General: dense, long sentences; table-caption codes (D, B0, w, B0/D) used as nouns in the prose; several terms never defined ("real-format steps," "pretraining-style data," "concentrability," "execution-consistency gate," "residual-on-frozen dynamics," "prior consistency," "joint grid," "broad head" in quotes, and C in (E, C, f, R, V, π) which is never named). The abstract's finding (i) is one 90-word sentence. The metaphors ("buys the axis," "coverage is bought in simulation," "order of effort") substitute for definitions. No figure in a 4-page draft with one page to spare.

Ten rewrites:

1. **L26** "we find that (i) the ability to operate under a new dynamics regime is set entirely by the planner's sampling prior: when the prior's domain randomization does not reach the regime, neither planning on the pretrained model nor dynamics adaptation from 45k real-format steps reduces falls (10 per 1000 steps for every prior whose randomization stops above 0.7× nominal torque), whereas a prior that covers it cuts falls fourfold with no real data;"
   → "we find that (i) which regimes the robot can operate in is decided by the prior's randomization, not by adaptation: at 0.4× torque, every prior randomized down to 0.7× or less falls about 10 times per 1000 steps, and neither planning nor adapting the dynamics on 45k logged steps changes that, while a prior randomized down to 0.3× falls 2.2 times per 1000 steps with no real data;"

2. **L32** "Our findings reverse the usual order of effort."
   → "Our findings assign the components different jobs than the pipeline assumes: coverage must come from pretraining, and real data is more useful for identifying the physics than for fitting the model."

3. **L30** "The implicit promise is that real-data adaptation plus planning handles conditions the simulation did not cover."
   → "SimDist is validated under nominal conditions; whether minutes of real data can also move the system into a dynamics regime the simulation never randomized over is untested. We test it." (Avoids the straw man; SimDist does not promise out-of-DR transfer.)

4. **L52** "the prior buys the axis it was randomized on."
   → "randomizing an axis buys robustness on that axis only, and costs performance elsewhere: nominal reward falls 22 % and falls at friction 0.15 double (1.0 → 2.2)."

5. **L66** "Data from the other prior degrade both updates by more than 2× relative to no update; data from the deployed prior never hurt and improve the nominal condition"
   → "Data collected by the other prior make both updates worse than no update (×2.15 for the dynamics, ×2.83 for the prior); data collected by the deployed prior leave the 0.4× condition unchanged (×1.00, ×1.12) and improve the nominal one"

6. **L72** "reduced falls at the true 0.4 from 1.49 to 1.24 ([0.62, 1.11]), whereas adapting it on on-policy data from the true condition raised them to 2.74 ([1.25, 2.51])"
   → "left falls at the true 0.4 unchanged (1.49 → 1.24; ratio 0.83, 95 % CI [0.62, 1.11]), whereas adapting it on on-policy data from the true condition raised them to 2.74 (ratio 1.84, CI [1.25, 2.51])"

7. **L72** "the fall-heavy saturated dynamics is learned at the expense of the frozen value head's calibration."
   → "We hypothesize that the adapted dynamics reproduces saturated, fall-prone latent trajectories that the frozen value head misreads; we have not tested this."

8. **L69** "Replaying logged transitions in the simulator identifies it nonetheless."
   → "The same logs do identify it when replayed through the simulator, because the deployment logs at 0.4× contain saturation events that the expert's training data do not."

9. **L81** "Real data is therefore best spent on identifying the physics, after which coverage is bought in simulation."
   → "Real data is therefore more valuable for identifying the actuator regime than for fitting the latent dynamics; once the regime is identified, coverage is obtained by retraining the prior in simulation."

10. **Table I caption / L45** "D: prior alone; B0: prior + planner on the pretrained model. w: lower bound of the prior's motor randomization."
    → Row labels "Prior alone" and "Prior + planner (pretrained model)"; column header "Lower bound w of the expert's motor randomization U(w, 1.1)"; a column "SimDist head" and "Broad head" so the fair comparison and the sweep are in one table; CIs in every cell. Delete D/B0 from the prose.

Also: L40 "45k real-format steps" → "45k steps logged in the same format the robot produces (proprioception, commands, executed actions)"; L32 "in line with the concentrability term" → add "(the ratio of the deploying policy's state distribution to the collecting policy's)."

---

## 6. Structure against the Stage-1 criteria

RSS Stage 1 asks reviewers to judge the problem statement, hypotheses and methodology. Against that:

- **Problem statement:** present (L30) but framed as a straw man about SimDist's "implicit promise." Reframe as a scoping question (what can the frozen-component adaptation recipe reach?).
- **Hypotheses:** the draft says the simulation predictions were pre-registered (L34, contribution 1) but lists none of them; only the hardware predictions appear (L75). The project's design docs contain the sim predictions, and at least two failed (planner beats prior-alone outside DR: not found; nominal cost ≤ 10 %: failed at −17/−22 %) and one claim was retracted (motor scale readable from history). A Stage-1 reviewer who is told "pre-registered" and then sees only the confirmed ones will assume the rest were dropped. Add a hypotheses table: prediction, pre-registered threshold, outcome, including failures.
- **Methodology / algorithm figure:** none. There is room (1 page). Needed: a one-panel figure showing the pipeline (which components frozen, where the real data enters under SimDist vs. under the recipe) and an algorithm box for the identifier (see §4).
- **Experimental protocol detail:** missing for the sim side: expert training (PPO, 1024 envs, 5000 iterations per the notes), seeds, what "falls per 1000 steps" means (termination + reset?), speed-command distribution, how the 45k steps are split across conditions and episodes, the adaptation recipe (3000 steps, dynamics only), and the identifier's parameters.
- **Hardware section (L75) as pre-registration:** half-credible. Credible parts: ground-truth cap at s × 23.7 Nm, four conditions, model list, 10 × 30 s runs, logged metrics, two numerical predictions (±0.15; "walks where SimDist falls or stands"). Placeholder parts: "Preliminary results will be inserted here (sessions in November)" is a placeholder in a document whose abstract promises "preliminary hardware results"; no definition of "walks" (speed-tracking threshold? distance?); "the planner's gain... is larger than in simulation" has no number and is unfalsifiable; no statement of robot count (one Go1?), battery control, run ordering, or the s = 0.6 identifiability issue (the identifier is weak above ≈0.7, so ±0.15 at 0.6 is near its limit); the "coupling test" adapts on 29.5 min collected by the narrow prior while the recipe says one minute. Either insert the numbers or label the section "Protocol (pre-registered)" and drop "preliminary results" from the abstract and contributions.

---

## 7. Predicted scores and the three changes that matter

| Criterion | Score (1–5) |
|---|---|
| Novelty | 2 |
| Soundness | 2 |
| Clarity | 2 |
| Significance | 3 |

**Justification.** The question is good and relevant to RSS (SimDist is an RSS 2026 paper; "what does its adaptation step actually buy?" is a question people will ask), which is why significance is a 3. Novelty is a 2 because the recipe is DROPO/SimOpt in SimDist's clothing and the coupling result is textbook; the one defensible new finding is the negative result on frozen-component adaptation outside the DR, and the draft buries it under a recipe. Soundness is a 2 because the headline numbers come from different seeds/runs and disagree by 2–3× (10.2 vs 3.8 vs 4.1; 2.2 vs 0.9), two CIs are omitted from the one row where they would contradict the text, two CIs in Sec. IV are mislabeled and one is non-significant yet reported as "reduced," the width sweep is single-seed with admitted seed variance equal to the effect, and step (4) of the recipe is confounded by the identifier's own bias. Clarity is a 2 because a reader cannot reconstruct the identifier, cannot find the hypotheses, cannot tell D from B0 without the caption, and has no figure. Predicted Stage-1 outcome: weak reject; a fixed version is a plausible weak accept because the core negative result is real and the hardware ground-truth idea is good.

**Three changes that would most raise the score.**

1. **One coherent set of numbers, with CIs everywhere.** Pick the fair comparison (SimDist's own head vs. the retrained heads) as the headline, put the width sweep and the head comparison in one table with CIs in every cell, run ≥3 expert seeds per width (or drop the sweep to w ∈ {none, 0.5, 0.3}), and delete or footnote the 3.8/3.2/0.9 numbers. Report the w = 0.9/0.7 CIs and rewrite "changes nothing" to what they show. Fix the two mislabeled CIs and call the identified-sim result non-significant.
2. **Make the title experiment the central table, and make the recipe a method.** Finish P10f (targeted 0.3–0.7 vs. blanket 0.3–1.1 vs. narrow) and add the missing control for step (4) (on-policy sim data at the true s = 0.4 vs. at the identified 0.5, on the same head), so that "identify, then cover" is tested rather than announced. Add an algorithm box with every identifier parameter and an automatic margin rule (posterior quantiles plus measured bias) in place of the hand-chosen 0.3–0.7. State the retraining cost per identified regime.
3. **Hypotheses table and figure; honest hardware section.** A table of all pre-registered predictions (sim and hardware) with thresholds and outcomes, including the two that failed and the retracted claim; a pipeline figure; and a hardware section titled as a protocol with operational success criteria, or with the actual numbers. Remove "preliminary hardware results" and "first hardware-relevant measurement" from the abstract and related work until data exist. Cite SimOpt and BayesSim and position against DROPO explicitly ("DROPO + SimDist, with the finding that...").
