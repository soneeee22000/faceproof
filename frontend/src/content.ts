// Structured content for the FaceProof landing page. Keeping copy and figures
// in one module keeps the section components presentational and lets every
// metric trace back to a single, auditable source.

export const LIVE_URL = "https://faceproof-102991984200.europe-west1.run.app";
export const GITHUB_URL = "https://github.com/soneeee22000/faceproof";
export const AUTHOR_NAME = "Pyae Sone (Seon)";
export const AUTHOR_URL = "https://github.com/soneeee22000";

/** Headline figures shown in the hero readout strip. */
export interface HeadlineStat {
  readonly label: string;
  readonly value: string;
  readonly caption: string;
}

export const HEADLINE_STATS: readonly HeadlineStat[] = [
  { label: "LFW ROC AUC", value: "0.9903", caption: "6 000-pair protocol" },
  { label: "Equal error rate", value: "2.22%", caption: "verification" },
  { label: "Anti-spoof ACER", value: "0.02%", caption: "ISO/IEC 30107-3" },
  { label: "Test suite", value: "62", caption: "ruff · mypy --strict" },
];

/** A gap in typical reference implementations, paired with FaceProof's answer. */
export interface ProblemGap {
  readonly index: string;
  readonly flaw: string;
  readonly flawDetail: string;
  readonly fix: string;
}

export const PROBLEM_GAPS: readonly ProblemGap[] = [
  {
    index: "A",
    flaw: "Guessed thresholds",
    flawDetail:
      "A hard-coded similarity cut-off copied from a blog post — never measured against real data.",
    fix: "The match threshold (0.2528) is read straight off the LFW ROC curve, at the accuracy-maximizing operating point.",
  },
  {
    index: "B",
    flaw: "No spoof handling",
    flawDetail:
      "A printout or a phone replay sails through a match-only pipeline as a perfect impostor.",
    fix: "A MobileNetV2 anti-spoofing model — transfer-learned on CelebA-Spoof — gates every verdict, with a Silent-Face baseline as fallback.",
  },
  {
    index: "C",
    flaw: "Self-graded demos",
    flawDetail:
      "Accuracy claimed on a handful of hand-picked images the author scored themselves.",
    fix: "Numbers come from standard protocols — LFW 6 000-pair, CelebA-Spoof — reproducible from committed raw scores.",
  },
];

/** One stage of the stateless verification pipeline. */
export interface PipelineStage {
  readonly step: string;
  readonly name: string;
  readonly summary: string;
  readonly tech: string;
}

export const PIPELINE_STAGES: readonly PipelineStage[] = [
  {
    step: "01",
    name: "Detect",
    summary:
      "Locate every face in the frame and return five facial landmarks per face.",
    tech: "InsightFace SCRFD",
  },
  {
    step: "02",
    name: "Align",
    summary:
      "Warp the landmarks onto the canonical 112×112 crop so pose stops being noise.",
    tech: "5-point similarity transform",
  },
  {
    step: "03",
    name: "Embed",
    summary:
      "Map the aligned face to a 512-d vector, L2-normalized so matching is a dot product.",
    tech: "ArcFace w600k_r50",
  },
  {
    step: "04",
    name: "Match",
    summary:
      "Score cosine similarity between the two embeddings against the calibrated threshold.",
    tech: "Cosine · threshold 0.2528",
  },
  {
    step: "05",
    name: "Liveness",
    summary:
      "Classify the selfie as a live face or a presentation attack — print, replay, mask.",
    tech: "MobileNetV2 anti-spoofing",
  },
  {
    step: "06",
    name: "Decide",
    summary:
      "Require match AND live. Emit a verdict with the reasons that produced it.",
    tech: "Explainable rule",
  },
];

/** A single readout in a metrics panel. */
export interface Metric {
  readonly label: string;
  readonly value: string;
  readonly note: string;
}

export interface MetricGroup {
  readonly index: string;
  readonly title: string;
  readonly dataset: string;
  readonly metrics: readonly Metric[];
  readonly footnote: string;
}

export const METRIC_GROUPS: readonly MetricGroup[] = [
  {
    index: "01",
    title: "Face verification",
    dataset: "LFW · 6 000-pair protocol",
    metrics: [
      { label: "ROC AUC", value: "0.9903", note: "ranking quality" },
      { label: "Equal error rate", value: "2.22%", note: "FAR = FRR" },
      { label: "Accuracy", value: "98.81%", note: "at operating point" },
      { label: "Threshold", value: "0.2528", note: "calibrated, cosine" },
    ],
    footnote:
      "Every figure is reproducible from the committed raw scores, or end-to-end on a free GPU.",
  },
  {
    index: "02",
    title: "Liveness / anti-spoofing",
    dataset: "CelebA-Spoof · 10 020-image held-out split",
    metrics: [
      { label: "ACER", value: "0.02%", note: "trained MobileNetV2" },
      { label: "APCER", value: "0.04%", note: "attacks passed" },
      { label: "BPCER", value: "0.00%", note: "live faces rejected" },
      { label: "Val accuracy", value: "99.90%", note: "67k-image subset" },
    ],
    footnote:
      "Scored with APCER / BPCER / ACER per ISO/IEC 30107-3, benchmarked against the Silent-Face baseline.",
  },
];

/** Technologies behind the system, grouped for the footer / stack list. */
export interface TechGroup {
  readonly category: string;
  readonly items: readonly string[];
}

export const TECH_STACK: readonly TechGroup[] = [
  {
    category: "Computer vision",
    items: ["InsightFace SCRFD", "ArcFace", "ONNX Runtime", "OpenCV"],
  },
  {
    category: "Training & eval",
    items: ["PyTorch", "torchvision", "NumPy", "scikit-learn"],
  },
  {
    category: "Service",
    items: ["FastAPI", "React", "TypeScript", "Vite"],
  },
  {
    category: "Delivery",
    items: ["Docker", "GCP Cloud Run", "GitHub Actions"],
  },
];
