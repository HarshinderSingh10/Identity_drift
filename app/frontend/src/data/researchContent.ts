/**
 * Central, typed research content.
 *
 * PRIMARY SOURCE: IEEE_major_project.pdf, "Longitudinal Embedding Drift in Facial
 * Representations: A Study of Temporal Stability and Verification Performance".
 * Every entry carries a `source` string (paper section, table, figure or page)
 * so it can be checked against the PDF. Text added only to help general readers
 * lives in `explainers` and is always rendered with the "Plain-language
 * explanation" label; it is never paper-derived.
 *
 * Numbers are stored as strings exactly as printed so no rounding or
 * reformatting can alter them.
 */

export interface Sourced {
  text: string
  /** Where in the paper this statement comes from. */
  source: string
}

/* -------------------------------------------------------------------------- */
/* Paper metadata                                                             */
/* -------------------------------------------------------------------------- */

export const paper = {
  title:
    'Longitudinal Embedding Drift in Facial Representations: A Study of Temporal Stability and Verification Performance',
  authors: ['Harshinder Singh', 'Gyan Sharma', 'Devansh Goel', 'Siddhant Nirwal'],
  department: 'Department of Computer Science and Engineering',
  institute: 'ABES Institute of Technology',
  location: 'Ghaziabad, India',
  indexTerms: [
    'Face recognition',
    'Face embeddings',
    'Longitudinal analysis',
    'Embedding drift',
    'ArcFace',
  ],
  /** Public path of the PDF (resolved with import.meta.env.BASE_URL). */
  pdfPath: 'paper/IEEE_major_project.pdf',
  source: 'Title block (p. 1)',
} as const

/** Abstract, verbatim from the paper (p. 1). */
export const abstract: Sourced = {
  source: 'Abstract (p. 1)',
  text: 'Face recognition systems typically evaluate whether two images belong to the same identity, but a different question is equally important: how stable is the numerical representation of the same person across longitudinal observations? We study this question using MORPH, a longitudinal face database, and fixed ArcFace-based embeddings. We construct an identity-disjoint train/validation/test protocol, measure embedding drift as 1 − cos(eᵢ, eⱼ), and relate drift to genuine verification outcomes. In this MORPH cohort, the same-person face embedding was not perfectly stable across time, and higher observed drift was associated with larger longitudinal age gaps. Identity-level drift was heterogeneous, indicating that representation change is not uniform across individuals. Verification performance remained strong at the fixed operating threshold in the held-out test split, but failure rates increased markedly for larger age gaps and for high-drift comparisons. We interpret these findings as observational evidence about representation stability and verification behavior, not as causal proof of aging effects or a validated re-enrollment policy. The study motivates future drift-aware re-enrollment research while emphasizing that adaptive re-enrollment remains a direction for future experimental validation.',
}

/** Short abstract summary for the landing page (condensed, no added claims). */
export const abstractSummary: Sourced = {
  source: 'Abstract (p. 1); Section I',
  text: 'Face recognition usually asks whether two images show the same person. This paper asks a different question: how stable is the numerical representation of the same person across longitudinal observations? Using the MORPH longitudinal database and fixed ArcFace-based embeddings, the authors measure embedding drift and relate it to genuine verification outcomes. They report observational evidence about representation stability and verification behavior, not causal proof of aging effects or a validated re-enrollment policy.',
}

/* -------------------------------------------------------------------------- */
/* Problem, motivation, question, objective                                   */
/* -------------------------------------------------------------------------- */

export const motivation: Sourced[] = [
  {
    source: 'Section I, paragraph 1',
    text: 'Face recognition is a mature research area, yet most evaluation protocols focus on matching two images at a single point in time or under controlled acquisition conditions.',
  },
  {
    source: 'Section I, paragraph 1',
    text: 'Longitudinal face data introduces a different problem: the same identity is observed repeatedly over time, and the face representation may drift as a function of age, acquisition conditions, and other covariates.',
  },
  {
    source: 'Section I, paragraph 2',
    text: 'A face-recognition system may still perform well overall while the underlying representation of the same identity changes measurably over time. The study therefore separates conventional recognition accuracy from representation stability.',
  },
]

export const researchQuestion: Sourced = {
  source: 'Section I, paragraph 3',
  text: 'How much does a person’s face embedding change across longitudinal observations, how does this embedding drift vary across individuals and longitudinal age gaps, and how is representation drift related to face-verification performance?',
}

export const objective: Sourced = {
  source: 'Section I, paragraph 2',
  text: 'The objective is not to claim a universal aging mechanism or a new recognition algorithm, but to characterize longitudinal embedding drift and its relationship to genuine verification outcomes.',
}

export const contributions: Sourced[] = [
  {
    source: 'Section I, paragraph 4',
    text: 'Characterize longitudinal drift in a large MORPH cohort using fixed, identity-disjoint train/validation/test splits.',
  },
  {
    source: 'Section I, paragraph 4',
    text: 'Quantify how observed drift changes across longitudinal age-gap groups.',
  },
  {
    source: 'Section I, paragraph 4',
    text: 'Evaluate the relationship between drift and genuine verification performance under a frozen threshold.',
  },
  {
    source: 'Section I, paragraph 4',
    text: 'Analyze identity-level heterogeneity in representative trajectories.',
  },
  {
    source: 'Section I, paragraph 4',
    text: 'Report an exploratory individualized stability model, showing that the first Identity Stability Index formulation did not add predictive information beyond current drift.',
  },
]

export const contributionsSummary: Sourced = {
  source: 'Section I, paragraph 4',
  text: 'Together, these findings provide a cautious empirical basis for future drift-aware re-enrollment work rather than a validated re-enrollment policy.',
}

/* -------------------------------------------------------------------------- */
/* Related work (Section II). `cites` are reference numbers.                  */
/* -------------------------------------------------------------------------- */

export interface RelatedWorkBlock extends Sourced {
  heading: string
  cites: number[]
}

export const relatedWork: RelatedWorkBlock[] = [
  {
    heading: 'Deep embedding models',
    cites: [1, 2, 3],
    source: 'Section II, paragraph 1',
    text: 'Face recognition systems have been transformed by deep embedding models such as FaceNet, CosFace, and ArcFace. These methods learn embeddings such that semantically similar faces are represented by nearby vectors, while different identities are separated in feature space. ArcFace remains a strong reference model for fixed-feature evaluations because it provides a stable backbone widely used in face-recognition research.',
  },
  {
    heading: 'Longitudinal datasets',
    cites: [4, 5],
    source: 'Section II, paragraph 2',
    text: 'Longitudinal face analysis is closely related to aging and template adaptation. MORPH is a classic longitudinal face dataset designed for age progression studies and repeated observations of the same identities over time. AgeDB is a widely used benchmark for age-invariant face recognition and age-gap evaluation. These datasets support analysis of how image appearance and model representations evolve as people age, but they do not fully isolate the causal effects of biological age from changes in lighting, pose, expression, camera, and image acquisition.',
  },
  {
    heading: 'Elapsed time and matching performance',
    cites: [6, 7],
    source: 'Section II, paragraph 3',
    text: 'A substantial body of recognition work investigates how elapsed time and age progression affect matching performance, including studies of age-invariant recognition and template aging. However, much of this literature focuses on same-vs-different identity decisions rather than the representational drift of the same identity itself.',
  },
]

export const researchGap: Sourced = {
  source: 'Section II, paragraphs 3–4',
  text: 'Instead of asking only whether two images match, this work quantifies the amount of change in the embedding of the same person over time and asks whether larger observed drift corresponds to increased verification failure. The observed relationship between drift and failure is statistical and empirical; it does not establish causality, and it does not yet justify a validated adaptive re-enrollment system. The study contributes a longitudinal representation-stability perspective rather than a direct application-oriented recognition system.',
}

/* -------------------------------------------------------------------------- */
/* Dataset, split, pairs                                                      */
/* -------------------------------------------------------------------------- */

export const studyFacts = {
  rawImages: '50,015',
  processedImages: '50,013',
  failedImages: '2',
  zeroFaceCases: '2',
  singleFaceCases: '50,013',
  multipleFaceCases: '0',
  rawIdentities: '19,033',
  eligiblePairs: '84,133',
  identitiesWithTwoOrMore: '13,119',
  embeddingModel: 'InsightFace buffalo_l model',
  embeddingDimension: '512',
  alignmentSize: '112 × 112',
  splitSeed: '42',
  impostorSeed: '42',
  impostorFarRule: '1%',
  similarityThreshold: '0.206770330667',
  driftBoundary: '0.793229669333',
  exploratoryTrajectoryThreshold: '0.361454629',
} as const

/** Paper Table I. */
export const datasetSummary = {
  source: 'Table I (p. 2)',
  columns: ['Dataset property', 'Value', 'Notes'],
  rows: [
    ['MORPH raw images', '50,015', 'Primary longitudinal dataset'],
    ['MORPH successfully processed', '50,013', '2 failed cases'],
    ['MORPH identities (raw audit)', '19,033', 'Approximate audit count'],
    ['AgeDB images', '12,240', 'Secondary audit dataset'],
    ['AgeDB identities', '440', 'Approximate audit count'],
    ['AgeDB protocol files', '4', '5-, 10-, 20-, 30-year'],
    ['Eligible MORPH longitudinal pairs', '84,133', 'Same identity, different observation'],
  ],
}

/** Paper Table II. */
export const splitSummary = {
  source: 'Table II (p. 2)',
  columns: ['Split', 'Identities', 'Images', 'Longitudinal pairs', 'Notes'],
  rows: [
    ['Train', '13,323', '35,040', '59,545', 'No identity overlap'],
    ['Validation', '2,855', '7,581', '12,303', 'No identity overlap'],
    ['Test', '2,855', '7,394', '12,289', 'No identity overlap'],
  ],
}

/** Section III-C, longitudinal coverage. */
export const coverageSummary = {
  source: 'Section III-C (pp. 2–3)',
  columns: ['Subset of identities', 'Identities'],
  rows: [
    ['At least two observations', '13,119'],
    ['At least three observations', '7,514'],
    ['At least four observations', '3,872'],
    ['At least five observations', '2,188'],
    ['At least three observations spanning at least five years', '326'],
    ['Spanning at least ten years (observation count not restated in the paper)', '145'],
    ['Spanning at least twenty years (observation count not restated in the paper)', '26'],
  ],
}

/* -------------------------------------------------------------------------- */
/* Pipeline (Fig. 1, p. 2)                                                    */
/* -------------------------------------------------------------------------- */

export interface PipelineStage {
  label: string
  /** Future-work stage drawn with a dashed outline. */
  future?: boolean
}

export const pipelineStages: PipelineStage[] = [
  { label: 'MORPH' },
  { label: 'Dataset Audit' },
  { label: 'Identity-Disjoint Split' },
  { label: 'Face Detection + Alignment' },
  { label: 'ArcFace / InsightFace' },
  { label: '512-D Embeddings' },
  { label: 'Longitudinal Pair Construction' },
  { label: 'Embedding Drift' },
  { label: 'Verification Analysis' },
  { label: 'Identity Stability Analysis' },
  { label: 'Future Adaptive Re-enrollment', future: true },
]

/* -------------------------------------------------------------------------- */
/* Reported results (Section V)                                               */
/* -------------------------------------------------------------------------- */

/** Section V-A. Values as printed. */
export const identityDriftSummary = {
  source: 'Section V-A (p. 3); Fig. 3',
  columns: ['Statistic', 'Value'],
  rows: [
    ['Mean', '0.2296'],
    ['Standard deviation', '0.0813'],
    ['Minimum', '0.0559'],
    ['25th percentile', '0.1818'],
    ['Median', '0.2180'],
    ['75th percentile', '0.2620'],
    ['90th percentile', '0.3118'],
    ['95th percentile', '0.3469'],
    ['99th percentile', '0.4653'],
    ['Maximum', '1.2184'],
  ],
}

/** Numeric twins of identityDriftSummary for charting (same values, SD excluded). */
export const identityDriftChart = [
  { label: 'Minimum', value: 0.0559 },
  { label: '25th percentile', value: 0.1818 },
  { label: 'Median', value: 0.218 },
  { label: 'Mean', value: 0.2296 },
  { label: '75th percentile', value: 0.262 },
  { label: '90th percentile', value: 0.3118 },
  { label: '95th percentile', value: 0.3469 },
  { label: '99th percentile', value: 0.4653 },
  { label: 'Maximum', value: 1.2184 },
]

/** Section V-B text: mean drift by age gap (approximate values as printed in the text). */
export const ageGapTextMeans = {
  source: 'Section V-B (p. 3); sample sizes from the n labels in Fig. 4 (p. 4)',
  columns: ['Age gap (years)', 'Mean drift (text, approx.)', 'n (Fig. 4 label)'],
  rows: [
    ['0–2', '0.236', '77,762'],
    ['3–4', '0.261', '4,676'],
    ['5–9', '0.305', '922'],
    ['10–19', '0.375', '660'],
    ['20+', '0.774', '113'],
  ],
}

/** Paper Table III. */
export const performanceSummary = {
  source: 'Table III (p. 4)',
  columns: ['Split', 'ROC-AUC', 'EER', 'FAR', 'FRR'],
  rows: [
    ['Train', '0.996767', '0.002625', '0.007598', '0.005250'],
    ['Validation', '0.996745', '0.002751', '0.010157', '0.005501'],
    ['Test', '0.996201', '0.003745', '0.008592', '0.007270'],
  ],
}

/** Paper Table IV (test split only). */
export const ageGapSummary = {
  source: 'Table IV (p. 4); Fig. 5',
  columns: ['Age gap (years)', 'Pairs', 'Mean drift', 'Acceptance', 'FRR'],
  rows: [
    ['0–2', '11,114', '0.2382', '99.93%', '0.07%'],
    ['3–4', '836', '0.2693', '99.76%', '0.24%'],
    ['5–9', '219', '0.3363', '93.61%', '6.39%'],
    ['10–19', '89', '0.3662', '88.76%', '11.24%'],
    ['20+', '31', '0.6681', '38.71%', '61.29%'],
  ],
}

/** Numeric twins of Table IV for charting. Must match ageGapSummary exactly. */
export const ageGapChart = [
  { label: '0–2', pairs: 11114, meanDrift: 0.2382, frrPercent: 0.07 },
  { label: '3–4', pairs: 836, meanDrift: 0.2693, frrPercent: 0.24 },
  { label: '5–9', pairs: 219, meanDrift: 0.3363, frrPercent: 6.39 },
  { label: '10–19', pairs: 89, meanDrift: 0.3662, frrPercent: 11.24 },
  { label: '20+', pairs: 31, meanDrift: 0.6681, frrPercent: 61.29 },
]

/** Paper Table V. */
export const isiSummary = {
  source: 'Table V (p. 5)',
  columns: ['Model', 'ROC-AUC', 'PR-AUC', 'Brier'],
  rows: [
    ['Current reference drift', '0.661054', '0.367662', '0.037012'],
    ['ISI', '0.661054', '0.367662', '0.037012'],
  ],
}

export const isiTerms: Sourced = {
  source: 'Section VII (p. 5)',
  text: 'The exploratory Identity Stability Index used current reference drift together with a small set of identity-specific historical terms: historical percentile, historical z-score, median deviation, consecutive drift, and recent slope.',
}

export const isiResult: Sourced = {
  source: 'Section VII (p. 5); Table V',
  text: 'The first formulation gave a negative result: the ISI score had ROC-AUC 0.661054 and PR-AUC 0.367662, exactly matching the current reference drift baseline, and the correlation between current drift and the ISI was 1.0. In this dataset and with this implementation, the ISI did not demonstrate independent predictive value beyond current drift.',
}

export const driftBinsFinding: Sourced = {
  source: 'Section V-E (p. 4); Fig. 6',
  text: 'The drift-bin analysis showed a consistent pattern: moderate drift values were still compatible with positive verification decisions, whereas sufficiently large drift values crossed the fixed verification operating boundary. The boundary (drift 0.7932) is a verification boundary, not a validated re-enrollment threshold.',
}

export const trajectoryFinding: Sourced = {
  source: 'Section V-F (p. 4); Fig. 7',
  text: 'The trajectory analysis showed heterogeneous identity-specific patterns: some identities had relatively stable drift across repeated observations, others showed increasing drift with age, and some had nonlinear or irregular movement. Candidate return/persistent events were identified using a training-derived exploratory threshold of 0.361454629; these are descriptive candidate events only, not evidence of a physiological mechanism or a production rule.',
}

/* -------------------------------------------------------------------------- */
/* Discussion, limitations, future work                                       */
/* -------------------------------------------------------------------------- */

export const discussion: Sourced[] = [
  {
    source: 'Section VI, paragraph 1',
    text: 'The main empirical result is not that embedding drift is a universal failure mode, but that it is measurable and varies with longitudinal separation. In the evaluated MORPH data, drift increased with larger age gaps, and larger age gaps were associated with higher FRR.',
  },
  {
    source: 'Section VI, paragraph 1',
    text: 'The effect is not uniform across identities: the distribution of identity-level mean drift was broad, and some identities remained stable while others accumulated larger representation changes over time.',
  },
  {
    source: 'Section VI, paragraph 2',
    text: 'A fixed recognition model can remain highly accurate overall even when substantial drift is present in some observations.',
  },
  {
    source: 'Section VI, paragraph 3',
    text: 'High drift does not automatically imply verification failure; moderate drift can still be compatible with successful matching under a fixed threshold.',
  },
]

export const limitations: Sourced[] = [
  {
    source: 'Section VIII, item 1',
    text: 'MORPH is observational; the data capture repeated observations but not controlled longitudinal interventions, so no causal effect of age on embedding drift can be inferred.',
  },
  {
    source: 'Section VIII, item 2',
    text: 'The age variable is filename-derived and is not a perfect biological-age measurement.',
  },
  {
    source: 'Section VIII, item 3',
    text: 'Observed drift may reflect multiple factors in addition to age, including illumination, pose, expression, facial hair, image quality, camera/device, compression, and appearance changes.',
  },
  {
    source: 'Section VIII, item 4',
    text: 'The 20+ year age-gap group contains only 31 test pairs.',
  },
  {
    source: 'Section VIII, item 5',
    text: 'The fixed ArcFace model is pretrained and not re-optimized for this longitudinal setting.',
  },
  {
    source: 'Section VIII, item 6',
    text: 'The first ISI formulation was exploratory and did not add independent predictive value.',
  },
  {
    source: 'Section VIII, item 7',
    text: 'The Stage 8 bootstrap methodology was declared, but the executed analysis did not perform the intended bootstrap, so confidence intervals for the ISI analysis should not be claimed.',
  },
  {
    source: 'Section VIII, item 8',
    text: 'AgeDB was audited for supplemental context but was not used as the main image-level longitudinal validation set.',
  },
  {
    source: 'Section VIII, item 9',
    text: 'Any re-enrollment policy remains speculative, and future work is required before any adaptive re-enrollment system can be considered validated.',
  },
]

export const futureWork: Sourced[] = [
  {
    source: 'Section IX, paragraph 2',
    text: 'Genuinely independent individualized stability models.',
  },
  { source: 'Section IX, paragraph 2', text: 'Identity-cluster bootstrap uncertainty.' },
  {
    source: 'Section IX, paragraph 2',
    text: 'Policy simulations comparing no re-enrollment, fixed-period re-enrollment, drift-triggered re-enrollment, and individualized re-enrollment.',
  },
]

export const conclusion: Sourced = {
  source: 'Section IX, paragraphs 1–2 (pp. 5–6)',
  text: 'The same person’s representation was not perfectly stable across repeated observations, drift was heterogeneous across identities, and larger longitudinal age gaps were associated with higher observed drift and lower verification acceptance in the evaluated sample. At the fixed verification threshold, held-out test performance remained strong overall, but long-term separations exhibited substantially greater verification failure rates. These findings support a temporal stability interpretation rather than a causal or policy claim.',
}

/* -------------------------------------------------------------------------- */
/* Plain-language explanations ADDED for general readers (not from the paper) */
/* -------------------------------------------------------------------------- */

export const explainers = {
  embedding:
    'An embedding is a list of numbers (here 512 of them) that a face-recognition model produces from a face image. Images of the same person should give similar lists.',
  cosine:
    'Cosine similarity compares the direction of two embeddings: a value near 1 means they point almost the same way. Drift turns that into a change score, where 0 means identical direction and larger values mean more change.',
  driftVersusError:
    'Drift measures how much the representation changed. Verification asks whether a similarity score clears a threshold. A pair can show noticeable drift and still be accepted, which is what the paper reports for moderate drift.',
  identityDisjoint:
    'In an identity-disjoint split, every person appears in only one of train, validation or test, so the test results are not helped by the model or thresholds having seen that person before.',
  farFrr:
    'FAR is the rate at which different-person (impostor) pairs are wrongly accepted. FRR is the rate at which same-person (genuine) pairs are wrongly rejected. A threshold trades one against the other.',
  frozenThreshold:
    'Selecting the threshold on the validation split and freezing it before testing means the test split cannot influence the operating point.',
  significance:
    'Overall accuracy answers “can the system tell these two images apart from other people?” Drift asks a separate question: “how far has this person’s numerical representation moved?” The paper reports that the two are related (larger age gaps coincide with more drift and more rejections) but not interchangeable (moderate drift is often still accepted).',
} as const
