# -*- coding: utf-8 -*-
"""Generate the full Response Letter for the PRL major revision.

Idempotent: always rebuilds from Response Letter_original.docx (created once
from the user's draft) and overwrites Response Letter.docx.

Structure follows the R->A->C response-letter template:
  Title -> salutation -> Summary of Revision (user's draft, integrated)
  -> List of Major Changes (to the AE) -> R1 (general + 9 x R/A/C)
  -> R2 note -> R3 (4 x R/A/C) -> closing.

Comment texts are verbatim quotes from revison1.txt. All numbers
cross-checked against cas-sc-template.tex.
"""
import os
from shutil import copyfile
from docx import Document
from docx.shared import Pt, Inches

BASE = r"d:\project ML"
SRC = os.path.join(BASE, "Response Letter.docx")
BAK = os.path.join(BASE, "Response Letter_original.docx")

if not os.path.exists(BAK):
    copyfile(SRC, BAK)

doc = Document(BAK)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)

paras = doc.paragraphs
assert len(paras) >= 3, "unexpected document structure"

def _fmt(run, size=11, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return run

def add_para(runs, indent=None, space_before=0, space_after=6):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if indent:
        pf.left_indent = Inches(indent)
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, kw in runs:
        _fmt(p.add_run(text), **kw)
    return p

def add_heading(text, size=13, space_before=14):
    return add_para([(text, dict(size=size, bold=True))],
                    space_before=space_before, space_after=4)

def add_comment(num, topic, comment):
    add_para([
        ("Comment %d (%s). " % (num, topic), dict(bold=True)),
        ("\u201c" + comment + "\u201d", dict(italic=True)),
    ], indent=0.25, space_before=8, space_after=2)

def add_response(text):
    add_para([("Response. ", dict(bold=True)), (text, {})],
             indent=0.25, space_after=2)

def add_change(loc):
    add_para([("Change location. ", dict(bold=True)), (loc, {})],
             indent=0.25, space_after=6)

# ---- document head: title (kept), salutation (inserted before summary) ----
title = paras[0]
title.text = "Response to Review"
for r in title.runs:
    _fmt(r, size=14, bold=True)

SALUTATION = (
    "Dear Associate Editor and Reviewers,\n"
    "We thank you for the careful reading and the constructive comments, which "
    "have substantially improved the manuscript. All comments have been "
    "addressed point by point: we first summarize the revisions, then list the "
    "major changes, and finally respond to each comment with the corresponding "
    "change location in the revised manuscript."
)
paras[1].insert_paragraph_before(SALUTATION)

# ---- "Summary of Revision:" heading + polished user paragraph ----
paras[1].text = "Summary of Revision"
for r in paras[1].runs:
    _fmt(r, size=13, bold=True)

SUMMARY = (
    "In this revision we substantially extend the empirical scope of the study "
    "with three new datasets: Fashion-MNIST and CIFAR-10 (image benchmarks) and "
    "Yeast (a markedly harder tabular dataset). The two image benchmarks were "
    "added to test whether the proposed framework applies beyond small tabular "
    "data. The results not only reproduce the phenomenon observed on small "
    "tabular datasets but also reveal a new finding: test accuracy differs "
    "significantly between the fully complementary regime (\u03b1 = 0) and the "
    "fully overlapping regime (\u03b1 = 1). Distill-S improves test accuracy by "
    "4.4 points on CIFAR-10 (0.788 vs. 0.744) and 1.1 points on Fashion-MNIST "
    "(0.923 vs. 0.912) in the fully complementary regime, confirming that "
    "complementary teacher data is a genuine source of benefit. To further "
    "examine whether the phenomenon is modality-dependent, we additionally "
    "include Yeast, a considerably harder tabular dataset, which exhibits the "
    "same pattern (+1.0 point test accuracy). Together with new test-set, "
    "calibration, teacher-quality, sensitivity and capacity analyses, these "
    "results support the conclusion that dataset difficulty \u2014 rather than "
    "modality \u2014 determines when same-architecture distillation helps."
)
paras[2].text = SUMMARY
for r in paras[2].runs:
    _fmt(r, size=11)

# ---- list of major changes (to the AE) ----
add_heading("List of Major Changes")
changes = [
    "New image-benchmark experiments (Fashion-MNIST and CIFAR-10): full \u03b1-sweep "
    "(6 values) \u00d7 20 random seeds under the identical protocol (a three-block "
    "VGG-style CNN with 32\u201364\u2013128 channels and batch normalization, identical "
    "architecture for teacher and student, early stopping on validation loss). New "
    "section \u201cImage Benchmarks\u201d (\u00a74.6). At \u03b1 = 0 the generalization gap "
    "decreases from 0.077 to 0.040 on Fashion-MNIST (\u0394 = \u22120.038, 95% CI "
    "[\u22120.040, \u22120.037]) and from 0.222 to 0.121 on CIFAR-10 (\u0394 = \u22120.101, "
    "CI [\u22120.106, \u22120.097]); test accuracy increases by 1.1 points (0.923 vs. "
    "0.912) and 4.4 points (0.788 vs. 0.744) \u2014 with half of its training data, the "
    "distilled student matches Baseline-Full (all Holm-adjusted p < 10\u207b\u2074, "
    "rank-biserial r = \u00b11.0); at \u03b1 = 1.0 no advantage survives.",
    "New hard tabular probe (Yeast, 1,484 samples \u00d7 8 features, 10 classes): "
    "reproduces the image-dataset pattern (test accuracy +1.0 points, p = 0.028; gap "
    "\u0394 = \u22120.027, Holm-adjusted p = 0.002, r = \u22120.90). New paragraph "
    "\u201cDifficulty, not modality\u201d in \u00a74.8.",
    "Primary evaluation moved to the held-out test set: test accuracy, negative "
    "log-likelihood (NLL) and 15-bin expected calibration error (ECE) are now reported "
    "for every student group (new section \u201cTest-Set Performance and Calibration\u201d, "
    "\u00a74.3); the generalization gap is retained only as a secondary mechanistic "
    "diagnostic.",
    "New section \u201cTeacher Quality and Teacher\u2013Student Agreement\u201d (\u00a74.4): "
    "mean teacher test accuracy, teacher ECE and soft-label entropy per dataset "
    "(calibration degrades with difficulty: ECE 0.021\u20130.047 on separable tabular data "
    "vs. 0.227 Glass / 0.167 CIFAR-10 / 0.133 Yeast), and teacher\u2013student prediction "
    "agreement on the student\u2019s own samples (\u226596% on separable datasets vs. 77% on "
    "Glass). The teacher\u2013student disagreement rate orders the \u03b1 = 0 distillation "
    "benefit across seven dataset-level points (Spearman \u03c1 = 0.786, p = 0.036).",
    "Statistical analysis upgraded throughout: paired Wilcoxon signed-rank tests over 20 "
    "seeds, bootstrap 95% confidence intervals (10,000 resamples) for mean paired "
    "differences, rank-biserial effect sizes, and Holm step-down family-wise correction "
    "(\u00a73.4 and all results tables); figures show \u00b11 standard deviation over seeds.",
    "New section \u201cHyperparameter Sensitivity and Teacher Capacity\u201d (\u00a74.5): "
    "label-smoothing \u03b5 \u2208 {0.05, 0.1, 0.2, 0.3}; KD T \u2208 {1, 2, 4} \u00d7 \u03bb "
    "\u2208 {0.3, 0.5, 0.7}; teacher-capacity ablation (hidden sizes (32,16), (64,32), "
    "(128,64)). Documents the underfitting confound at high T/\u03bb (Glass, T = 2, "
    "\u03bb = 0.5: gap 0.108 with test accuracy 0.564 vs. 0.645 at T = 1).",
    "Data-construction details clarified (\u00a73.2): the feasibility constraint \u03b2 \u2265 "
    "\u03b1(1\u2212\u03b2) is now stated explicitly, asymmetric regimes (\u03b2 \u2260 0.5 with "
    "\u03b1 > 0) are excluded by design, the seeded sampling protocol is described, and a "
    "class-probability alignment issue is documented and fixed (teacher probabilities "
    "re-ordered via the teacher\u2019s classes_ attribute). All results were regenerated "
    "with the corrected pipeline.",
    "Related work updated: HASKD and MixSKD are now cited and discussed in an explicit "
    "protocol-positioning paragraph; reference list corrected (venue/booktitle of "
    "Furlanello et al. and Zhu et al.; UCI repository entry completed).",
    "Presentation fixes: citation style inconsistencies (e.g., \u201csettingsHinton\u201d, "
    "duplicated author\u2013year citations, the malformed \u201cStanton et al.\u201d entry), "
    "spacing around citations, the corresponding author\u2019s e-mail domain, and the "
    "typographical errors pointed out by the reviewers.",
]
for i, c in enumerate(changes, 1):
    add_para([("%d. " % i, dict(bold=True)), (c, {})])

# ---- Reviewer 1 ----
add_heading("Response to Reviewer #1")
add_para(
    "We thank Reviewer #1 for the encouraging assessment of our question as "
    "\u201crelevant and underexplored\u201d and for the detailed, constructive comments. "
    "The general assessment and each comment are quoted below, followed by our "
    "response and the corresponding change location."
)

R1_GENERAL = (
    "Overall, this manuscript studies how the overlap between teacher and student training "
    "sets affects knowledge distillation, and proposes an \u03b1-overlap framework to vary the "
    "fraction of shared samples. The topic is relevant and underexplored, especially because "
    "most KD studies implicitly assume that teacher and student are trained on the same or "
    "highly overlapping data. The paper is clearly organized, and the experimental design is "
    "easy to follow. The results suggest that complementary partitioning may reduce the "
    "student\u2019s generalization gap, with some dataset-dependent benefits over label "
    "smoothing. However, for Pattern Recognition Letters, the current version still has "
    "several important limitations. The empirical scope is relatively narrow, the claimed "
    "contribution is closer to an experimental observation than a new method or theoretical "
    "advance, and the analysis does not yet sufficiently demonstrate when, why, and to what "
    "extent complementary KD improves useful predictive performance. Therefore, I believe "
    "the manuscript requires substantial revision before it can be considered for publication."
)
add_para([("General assessment. ", dict(bold=True)),
          ("\u201c" + R1_GENERAL + "\u201d", dict(italic=True))],
         indent=0.25, space_before=8, space_after=2)
add_para([("Response. ", dict(bold=True)),
          ("We thank the reviewer for this accurate characterization of the original "
           "manuscript. The revision is designed precisely around the limitations "
           "identified: narrow empirical scope (Comments 2\u20133), usefulness of predictive "
           "performance (Comments 3\u20134), positioning (Comment 1), and mechanism "
           "(Comment 8); in addition, the statistics, implementation, and presentation are "
           "strengthened (Comments 5\u20137, 9).", {})],
         indent=0.25, space_after=6)

R1 = [
    ("novelty positioning",
     """The novelty and contribution should be better positioned. The proposed \u03b1-overlap framework is useful as an experimental protocol, but it is essentially a controlled data-partitioning scheme rather than a new knowledge distillation method. The manuscript should more clearly distinguish between methodological novelty and empirical observation, and explain how the proposed framework advances the KD literature beyond existing studies on data heterogeneity, self-distillation, federated distillation, and regularization by soft labels.""",
     "We agree that the original framing did not sufficiently distinguish our contribution. "
     "The revision clarifies that our contribution is not a new distillation loss but a "
     "controlled experimental framework: the \u03b1-overlap protocol isolates the single factor "
     "of teacher\u2013student data complementarity while holding architecture, capacity, "
     "optimization and data volume fixed. We have rewritten the contributions and added a "
     "protocol-positioning paragraph in Related Work that explicitly differentiates our "
     "setting from the four lines of work the reviewer lists: studies of data heterogeneity "
     "vary partitions but do not control the overlap ratio \u03b1 itself; self-distillation "
     "and federated distillation fix the teacher\u2013student data relationship by "
     "construction; and soft-label regularization (label smoothing) manipulates the targets "
     "without any teacher data. Within this controlled framework we characterize when "
     "same-architecture distillation helps \u2014 complementary teacher data, with the "
     "benefit increasing in dataset difficulty \u2014 and provide an agreement-based "
     "mechanism analysis. We now also cite HASKD and MixSKD and explain that neither "
     "studies the data-overlap axis.",
     "\u00a71 (Contributions), \u00a72 (Related Work)."),
    ("dataset scope and generality",
     """The experimental evidence is too limited for the current claims. The study uses only four relatively small benchmark datasets, including Wine, BreastCancer, Glass, and Digits. Several of these datasets are very small, and the observed effects may be highly sensitive to random splits, model capacity, and class imbalance. For a pattern recognition journal, stronger evidence on more representative datasets is expected, such as CIFAR-10/100, Fashion-MNIST, SVHN, or other image recognition benchmarks. Without such experiments, it is difficult to judge whether the conclusions generalize beyond small-scale tabular or low-dimensional settings.""",
     "We fully agree and have added two image benchmarks from the reviewer\u2019s list "
     "(Fashion-MNIST and CIFAR-10), each with a full \u03b1-sweep (6 values) \u00d7 20 seeds "
     "under the identical protocol (a three-block VGG-style CNN, identical architecture for "
     "teacher and student, early stopping on validation loss). The phenomenon "
     "reproduces on vision data with a larger effect size: at \u03b1 = 0 the generalization "
     "gap decreases from 0.077 to 0.040 (FMNIST) and from 0.222 to 0.121 (CIFAR-10), and "
     "test accuracy increases by 1.1 and 4.4 points respectively \u2014 with half of its "
     "training data, the distilled student matches Baseline-Full (all Holm-adjusted "
     "p < 10\u207b\u2074, r = \u00b11.0); at \u03b1 = 1.0 no advantage remains. The "
     "sensitivity concerns are addressed directly: random splits by 20 paired seeds per "
     "configuration; model capacity by a teacher-capacity ablation over three sizes "
     "(Comment 4); class imbalance by documenting that class composition is balanced only "
     "in expectation rather than by exact stratification (\u00a73.2). We additionally "
     "included Yeast as a hard tabular probe (\u00a74.8, \u201cDifficulty, not modality\u201d). "
     "SVHN and CIFAR-100 are excellent further candidates; we now note them explicitly as "
     "natural extensions, along with the intended scope (same-architecture, "
     "data-constrained distillation) and the limitation that larger-scale architectures are "
     "left to future work.",
     "\u00a73.1 (Dataset), \u00a74.6 (Image Benchmarks, new), \u00a74.8."),
    ("evaluation of actual predictive performance",
     """The paper focuses heavily on the generalization gap, but the actual predictive performance is insufficiently discussed. Reducing the train-test gap does not necessarily imply a practically better model. In several places, the manuscript notes that accuracy gains are close to zero, yet the central argument emphasizes regularization benefits. The authors should report and discuss test accuracy, validation accuracy, calibration metrics, and possibly negative log-likelihood alongside the generalization gap. This would help clarify whether complementary KD improves useful performance or merely reduces overfitting by lowering training accuracy.""",
     "All headline claims are now supported by held-out test metrics \u2014 test accuracy, "
     "NLL, and 15-bin ECE \u2014 reported for every student group in the new section "
     "\u201cTest-Set Performance and Calibration\u201d. On CIFAR-10 at \u03b1 = 0, "
     "Distill-S improves test accuracy by 4.4 points (0.788 vs. 0.744) and reduces NLL from "
     "0.772 to 0.665; on Fashion-MNIST, accuracy rises by 1.1 points (0.923 vs. 0.912) and "
     "NLL improves from 0.257 to 0.230; on the tabular datasets, the distilled student "
     "improves NLL on all four. The gains are therefore no longer \u201cclose to zero\u201d: "
     "on the harder datasets the benefit is now demonstrated on proper predictive metrics, "
     "and the generalization gap is retained only as a secondary mechanistic indicator. We "
     "also report calibration honestly: KD yields better-calibrated students than label "
     "smoothing on the tabular datasets and FMNIST, whereas on CIFAR-10 label smoothing "
     "achieves slightly lower ECE (0.049 vs. 0.068). On the tabular datasets the accuracy "
     "differences against Baseline-Half remain small and not significant after Holm "
     "correction --- the accuracy gains are a feature of the harder (image and Yeast) "
     "benchmarks.",
     "\u00a73.4 (Evaluation Metrics), \u00a74.3 (Test-Set Performance and Calibration, new)."),
    ("teacher quality and capacity",
     """Teacher quality and teacher-student capacity effects are not adequately analyzed. The effectiveness of distillation depends strongly on teacher accuracy, calibration, confidence, and capacity relative to the student. The current manuscript does not sufficiently examine whether the observed benefits arise from complementary data, from teacher regularization, from reduced memorization, or from specific teacher errors. Additional experiments with different teacher/student architectures, capacities, and teacher accuracies would make the conclusions more convincing.""",
     "We added a dedicated analysis reporting mean teacher test accuracy, teacher ECE "
     "(calibration) and soft-label entropy for every dataset and partition \u2014 all three "
     "degrade with task difficulty (ECE 0.021\u20130.047 on the separable tabular datasets "
     "vs. 0.227 on Glass, 0.167 on CIFAR-10 and 0.133 on Yeast) \u2014 together with a "
     "teacher-capacity ablation over three teacher sizes. Teacher quality varies "
     "substantially across datasets (0.965\u20130.970 on Wine/Breast Cancer/Digits vs. 0.662 "
     "on Glass), and the new agreement analysis shows this variation is informative rather "
     "than confounding: on the separable datasets the student reproduces the teacher\u2019s "
     "label on \u226596% of its own training samples, so soft labels act mostly as confidence "
     "smoothing, whereas on Glass agreement is only 77%. Regarding the source of the "
     "benefit, two results speak directly: (i) at \u03b1 = 1.0 (teacher trained on identical "
     "data) none of the advantages survive, and on CIFAR-10 distilled students fall "
     "slightly below Baseline-Half \u2014 so the benefit comes from complementary data, not "
     "from having a teacher per se; (ii) the teacher\u2013student disagreement rate on the "
     "student\u2019s own samples orders the \u03b1 = 0 benefit across seven dataset points "
     "(\u03c1 = 0.786, p = 0.036). The capacity ablation shows that teacher accuracy is "
     "nearly unchanged across sizes (Glass 0.654\u20130.662; Breast Cancer \u22480.970) and "
     "the student outcome does not qualitatively alter.",
     "\u00a74.4 (Teacher Quality and Teacher\u2013Student Agreement, new), "
     "\u00a74.5 (teacher-capacity ablation)."),
    ("fairness of the label-smoothing comparison",
     """The comparison with label smoothing is useful but incomplete. Since the manuscript argues that complementary KD can provide regularization beyond generic soft-target smoothing, the label smoothing baseline should be tuned more carefully. It is unclear whether the smoothing coefficient was optimized, whether temperature scaling in KD was tuned comparably, and whether both methods were given equal validation-based hyperparameter selection. A fairer comparison should include sensitivity analyses over label smoothing strength, distillation temperature, and the hard/soft loss mixing coefficient \u03bb.""",
     "We now sweep exactly the three quantities the reviewer lists: the "
     "label-smoothing strength \u03b5 \u2208 {0.05, 0.1, 0.2, 0.3}, the distillation "
     "temperature T \u2208 {1, 2, 4}, and the mixing coefficient \u03bb \u2208 {0.3, 0.5, "
     "0.7}. All groups share an identical validation-based model-selection protocol (early "
     "stopping on validation loss; the best-validation checkpoint is used for test "
     "evaluation). The LS baseline is stable across \u03b5 \u2014 on Digits, LS remains the "
     "stronger regularizer at every \u03b5 (gap 0.026\u20130.029), so the Digits result is "
     "not an artifact of \u03b5 = 0.1. The T\u00d7\u03bb grid also documents an important "
     "confound that we now analyze explicitly: large T and \u03bb produce an apparent "
     "reduction of the generalization gap while test accuracy and NLL degrade sharply "
     "(Glass, T = 2, \u03bb = 0.5: gap 0.108 with accuracy 0.564 vs. 0.645 at T = 1), i.e., "
     "student underfitting masquerading as improved generalization. Our default "
     "configuration (T = 1, \u03bb = 0.5) is at the predictive-performance optimum of the "
     "swept grid. We also report and discuss the cases where label smoothing outperforms "
     "Distill-S (Digits, where the teacher\u2019s soft labels are noisy).",
     "\u00a74.5 (Hyperparameter Sensitivity and Teacher Capacity, new)."),
    ("statistical analysis",
     """The statistical analysis needs more detail. The manuscript reports Wilcoxon signed-rank tests and p-values, but should provide confidence intervals or effect sizes in addition to significance markers. Given the small datasets and multiple comparisons across \u03b1, \u03b2, and datasets, the authors should also discuss whether any correction for multiple testing was applied. Reporting mean \u00b1 standard deviation over seeds for all key metrics would improve transparency.""",
     "All experiments now use 20 random seeds. We report paired Wilcoxon signed-rank tests "
     "(seeds as pairs), bootstrap 95% confidence intervals for the mean paired difference "
     "(10,000 resamples), the rank-biserial correlation as an effect size, and Holm "
     "step-down correction for multiple testing. These appear in every results table; all "
     "significance claims in the revised manuscript are Holm-adjusted, and figures show "
     "\u00b11 standard deviation over the 20 seeds.",
     "\u00a73.4 (Evaluation Metrics), all results tables in \u00a74."),
    ("\u03b1-construction details and reproducibility",
     """The definition and implementation of \u03b1-overlap require clarification. The notation defines \u03b1=(\u2223D_S\u2229D_T\u2223)/(\u2223D_S\u2223), but when \u03b2\u22600.5, the teacher and student set sizes differ. The feasible range and construction procedure for overlap may not be symmetric and may impose constraints depending on \u03b2. The manuscript should explicitly describe how overlapping and non-overlapping samples are selected under different \u03b2 values, and whether stratification by class is maintained.""",
     "\u00a73.2 now addresses each point. (i) Feasibility: since \u03b1 is defined relative to "
     "|D_S|, the design requires |D_T| \u2265 \u03b1|D_S|, i.e., \u03b2 \u2265 \u03b1(1\u2212\u03b2); "
     "the \u03b2-dependence of the feasible range is exactly the asymmetry the reviewer "
     "anticipates, and we now state that asymmetric regimes (\u03b2 \u2260 0.5 with \u03b1 > 0) "
     "are excluded by design \u2014 the \u03b1-sweep fixes \u03b2 = 0.5 and the \u03b2-sweep "
     "fixes \u03b1 = 0, so each sweep varies one factor in a symmetric regime. (ii) "
     "Selection: D_S is the tail of a seeded random permutation of the pool, and the "
     "non-overlapping part of D_T is drawn from the complementary pool half with an "
     "independent, seed-and-\u03b1-dependent permutation, making the construction exactly "
     "reproducible. (iii) Stratification: class composition is balanced only in expectation "
     "rather than by exact stratification, and no class is excluded from either partition; "
     "we now state this openly. In addition, we document a class-probability alignment fix "
     "in the distillation loss (teacher probabilities are now mapped onto the global class "
     "index set via the teacher\u2019s fitted classes_ attribute, so soft-label columns "
     "remain aligned even when a rare class is absent from a partition). All reported "
     "results were regenerated with the corrected pipeline; the qualitative conclusions "
     "are unchanged.",
     "\u00a73.2 (Experimental Design)."),
    ("mechanism",
     """The mechanism behind the observed effect remains underdeveloped. The manuscript suggests that complementary data may allow the teacher to provide inter-class structure not directly observed by the student, but this remains mostly speculative. The separability analysis using LDA and 1-NN is only exploratory and does not fully explain the results, especially the Digits case where label smoothing outperforms KD. Additional analyses, such as teacher confidence distributions, class-wise errors, calibration, entropy of soft labels, or agreement between teacher and student on overlapping versus non-overlapping regions, would strengthen the mechanistic interpretation.""",
     "We followed the reviewer\u2019s central suggestion \u2014 agreement between teacher and "
     "student on the student\u2019s own (overlapping) samples \u2014 and it substantially "
     "strengthens the interpretation: teacher\u2013student agreement is \u226596% "
     "on the separable datasets (soft labels act mostly as confidence smoothing) but only "
     "77% on Glass, i.e., roughly a quarter of the teacher\u2019s soft labels there encode "
     "inter-class structure the student has not recovered from its own data; this gradient "
     "tracks the benefit pattern. We additionally report the reviewer\u2019s suggested "
     "teacher-calibration and soft-label-entropy statistics: both degrade with "
     "task difficulty \u2014 teacher ECE rises from 0.021\u20130.047 on separable tabular "
     "data to 0.227 on Glass and 0.167 on CIFAR-10, and soft-label entropy from 0.05\u2013"
     "0.12 to 0.46 on Glass and 0.77 on Yeast \u2014 indicating that teachers on harder "
     "tasks encode more distributional information in their soft labels. The new "
     "\u201cDifficulty, not modality\u201d analysis "
     "(\u00a74.8) quantifies the disagreement\u2013benefit relation cross-dataset: the "
     "teacher\u2013student disagreement rate "
     "orders the \u03b1 = 0 benefit across all seven dataset-level points (Yeast 19.6% "
     "falls between CIFAR-10 13.3% and Glass 23.0%; Spearman \u03c1 = 0.786, p = 0.036). "
     "The Digits case the reviewer highlights is now analyzed explicitly: Digits is highly "
     "separable, LS remains the stronger regularizer at every \u03b5 (\u00a74.5), and the "
     "outcome there depends on teacher soft-label quality \u2014 a factor the overlap "
     "mechanism alone does not capture. Following the reviewer\u2019s assessment of the "
     "LDA/1-NN analysis, it is retained as motivation but demoted to exploratory status, "
     "and we state openly that the current experiments do not provide a complete "
     "theoretical explanation. In the future-work section we outline further moderators of "
     "the KD--LS tradeoff (class-count and calibration effects) and propose "
     "formalizing dataset difficulty as the disagreement rate of a trained model, i.e., a "
     "model-relative rather than intrinsic quantity.",
     "\u00a74.4 (agreement and teacher statistics), \u00a74.8 (Difficulty, not modality), "
     "\u00a75 (Discussion, future work)."),
    ("presentation",
     """The presentation would benefit from careful polishing. Several citation and reference formatting issues can be observed. For instance, there are missing spaces between the main text and citations, such as \u201csettingsHinton\u201d. Some in-text citations appear to repeat author names, such as \u201cHinton et al. Hinton et al. (2015)\u201d and \u201cZhu et al. Zhu et al. (2021)\u201d; another citation appears as \u201cStanton et al. Stanton, Izmailov, Kirichenko, Alemi and Wilson (2021)\u201d, which does not follow a standard author-year citation format. In addition, some reference entries appear incomplete or incorrectly punctuated, e.g., \u201cKelly, M., Longjohn, R., Nottingham, K., .\u201d. The manuscript should be carefully proofread for citation formatting, reference completeness, mathematical notation consistency, table/figure readability, and grammar.""",
     "All listed issues have been fixed: the missing space in \u201csettingsHinton\u201d; the "
     "duplicated author\u2013year citations (\u201cHinton et al. Hinton et al. (2015)\u201d, "
     "\u201cZhu et al. Zhu et al. (2021)\u201d) and the malformed \u201cStanton et al.\u201d "
     "citation now use consistent author-year commands; the incomplete UCI repository entry "
     "\u201cKelly, M., Longjohn, R., Nottingham, K., .\u201d is completed with year and "
     "howpublished fields. The manuscript has additionally been proofread for mathematical "
     "notation consistency, table/figure readability, and grammar, and the corresponding "
     "author\u2019s e-mail domain has been corrected. We thank the reviewer for the careful "
     "reading.",
     "Throughout; references list in \u00a78."),
]
for i, (topic, comment, resp, loc) in enumerate(R1, 1):
    add_comment(i, topic, comment)
    add_response(resp)
    add_change(loc)

add_para([("Response to the summary. ", dict(bold=True)),
          ("The reviewer\u2019s closing summary emphasizes four priorities \u2014 stronger "
           "pattern-recognition benchmarks, actual predictive performance in addition to "
           "generalization gaps, strengthened statistical analysis, and clarification of "
           "the mechanism and implementation of the overlap framework. These correspond "
           "respectively to Comments 2\u20133, 3\u20134, 6, and 7\u20138 above.", {})],
         indent=0.25, space_before=4, space_after=6)

add_para(
    "Reviewer #2 did not return a report for this round; there are accordingly "
    "no comments to address.", space_before=8
)

# ---- Reviewer 3 ----
add_heading("Response to Reviewer #3")
add_para(
    "We are grateful to Reviewer #3 for the positive assessment (well-organized "
    "manuscript, clear motivation, an interesting and underexplored question, "
    "useful controls, and reliability through multiple seeds and paired tests) "
    "and for the four concrete weaknesses, which are quoted below followed by our "
    "response and the corresponding change location."
)

R3 = [
    ("primary evaluation on a test set",
     """The held-out test set is apparently not used for the main evaluation, although an 80/20 train-test split is introduced. Since the validation set is also used for early stopping, conclusions about generalization should primarily be supported by test accuracy/test loss rather than the validation gap.""",
     "We agree. In the revision, all headline claims are supported by held-out test "
     "accuracy, test loss (NLL) and ECE; e.g., on CIFAR-10 at \u03b1 = 0, Distill-S gains 4.4 "
     "points of test accuracy (0.788 vs. 0.744) and reduces NLL from 0.772 to 0.665. The "
     "generalization gap is retained only as a complementary mechanistic indicator.",
     "\u00a73.4, \u00a74.3 (Test-Set Performance and Calibration, new)."),
    ("influence of teacher quality",
     """Teacher quality is insufficiently analyzed. The paper does not systematically report teacher accuracy, calibration, predictive entropy, or teacher-student performance gaps. Without these results, it is difficult to separate the effect of data complementarity from differences in teacher quality induced by different data partitions.""",
     "We added a dedicated teacher-quality analysis that now systematically "
     "reports, for every dataset, the four quantities the reviewer lists: teacher "
     "accuracy, calibration (teacher ECE), predictive entropy (mean soft-label entropy), "
     "and the teacher\u2013student performance relationship (on Yeast the student, 0.593, "
     "again exceeds its teacher, 0.569). Teacher accuracy, ECE and entropy all track task "
     "difficulty (ECE 0.021\u20130.047 on the separable tabular datasets vs. 0.227 on "
     "Glass). Crucially, the new agreement analysis separates the two effects the reviewer "
     "is concerned about: the teacher\u2013student disagreement rate on the student\u2019s "
     "own samples orders the \u03b1 = 0 distillation benefit across seven dataset points "
     "(\u03c1 = 0.786, p = 0.036), so the benefit can be attributed to the complementary "
     "information content of the partition rather than to teacher quality alone. "
     "Consistently, at \u03b1 = 1.0 (identical teacher data) no advantage survives. A "
     "teacher-capacity ablation over three sizes shows the conclusions are "
     "robust to teacher capacity.",
     "\u00a74.4 (Teacher Quality and Teacher\u2013Student Agreement, new), \u00a74.5."),
    ("gap reduction vs. true generalization",
     """The central evidence for improved generalization is not sufficiently convincing. The main metric is the training-validation accuracy gap. A smaller gap can simply result from lower training accuracy or slower optimization, rather than better generalization. Indeed, the paper explicitly reports no consistent improvement in best validation accuracy. Therefore, the current evidence supports a regularization/underfitting effect more directly than an actual improvement in predictive generalization.""",
     "We agree this is possible in general and now address it directly. First, the headline "
     "evidence is now test-set accuracy and NLL, which improve on the new benchmarks "
     "(CIFAR-10: +4.4 points accuracy, NLL 0.772 \u2192 0.665; FMNIST: +1.1 points, NLL "
     "0.257 \u2192 0.230) \u2014 indeed, with half of its training data the distilled "
     "student matches Baseline-Full \u2014 so the benefit no longer rests on the gap metric. "
     "Second, the new T\u00d7\u03bb grid demonstrates exactly the failure mode the reviewer "
     "worries about: at high temperature and weight the gap shrinks while test accuracy "
     "collapses (Glass, T = 2, \u03bb = 0.5: gap 0.108, accuracy 0.564 vs. 0.645 at T = 1), "
     "and our default configuration (T = 1, \u03bb = 0.5) is at the predictive-performance "
     "optimum of the swept grid. The underfitting confound is now documented, and the gap "
     "is demoted to a mechanistic diagnostic.",
     "\u00a73.4, \u00a74.3 (test metrics), \u00a74.5 (T\u00d7\u03bb grid)."),
    ("related-work coverage",
     """In the section of \u201c2. Related Work\u201d, the related-work coverage is rather limited. This paper should further include the discussion of \u201cHierarchical Self-supervised Augmented Knowledge Distillation\u201d and \u201cMixSKD: Self-Knowledge Distillation from Mixup for Image Recognition\u201d to make the analysis more comprehensive.""",
     "Following the reviewer\u2019s suggestion, we now cite and discuss both papers (HASKD "
     "and MixSKD) in Related Work, with a positioning paragraph explaining how our protocol "
     "differs: HASKD studies hierarchical and augmented-relational knowledge transfer, and "
     "MixSKD derives self-distillation targets from mixed samples, whereas we control the "
     "teacher\u2013student data-overlap axis itself.",
     "\u00a72 (Related Work)."),
]
for i, (topic, comment, resp, loc) in enumerate(R3, 1):
    add_comment(i, topic, comment)
    add_response(resp)
    add_change(loc)

# ---- closing ----
add_para(
    "We believe the manuscript is substantially stronger as a result of this "
    "revision, and we thank the Associate Editor and the reviewers again for "
    "their time and constructive input.", space_before=12
)
add_para("Sincerely,\nThe Authors", space_before=6)

doc.save(SRC)
print("saved:", SRC)
