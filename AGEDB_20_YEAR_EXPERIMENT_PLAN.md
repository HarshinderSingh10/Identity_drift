# AgeDB 20-Year Experiment Plan and Preprocessing Compatibility Audit

**Scope and boundary:** Read-only audit of `data\raw\AgeDB_Images`, `data\raw\AgeDB_protocols`, and the project code/configuration directly needed to describe the MORPH preprocessing pipeline. No embeddings, model inference, similarity scores, verification metrics, threshold selection, or experiments were run. Dataset files were not extracted or changed.

**Status terms:** **Verified** means directly observed in the project files, protocol/archive contents, or inspected project code. **Inferred** means supported by observed patterns or non-authoritative implementation code, but not established by the dataset authors’ protocol specification. **Unresolved** means the available evidence is insufficient.

## 1. Executive summary and go/no-go recommendation

- **CONDITIONAL GO for a future, separately approved external verification evaluation; NO-GO to begin embedding extraction now.** **Verified:** all 6,000 pair records in the 20-year MAT protocol are structurally well-formed, all 5,268 unique endpoint JPGs and matching `.pts` files are present in the archive, and every referenced JPG decoded successfully in the read-only audit. Ten folds of 600 rows can be preserved as stored.
- **Inferred:** `annot=1` is intended as same-identity and `annot=-1` as different-identity. This interpretation agrees with the data patterns and a public InsightFace-derived converter, but the MAT fields do not define the labels and no author-hosted protocol specification confirming them was located.
- **Unresolved:** the current MORPH source identifies the `buffalo_l` pack, but does not pin the exact resolved ONNX weight files or hashes. The installed InsightFace version, model-internal pixel scaling, exact MORPH threshold/score provenance, and authoritative global identity-token semantics also need confirmation.
- **NO-GO for identity-level AgeDB drift analysis for now.** Do not form longitudinal trajectories by globally grouping on the first filename token until its identity scope and stability are independently documented or validated.

## 2. Verified protocol structure and fold integrity

**Verified — structure:** `data\raw\AgeDB_protocols\03_FINAL_protocol_20_years.mat` is a MATLAB 5 MAT file with top-level variable `splits`, a `10 x 1` cell array. Each cell contains a `1 x 1` struct with fields `pairs` and `annot`. `pairs` is a `2 x 600` cell array; each endpoint is a `1 x 1` struct with fields `name` and `age`. `annot` is a `1 x 600` numeric array (stored as 16-bit integers). Every split has 600 records, for 6,000 total: 3,000 raw annotation values `1` and 3,000 values `-1`.

| Fold | Pair rows | Repeated unordered-pair rows within fold |
|---:|---:|---:|
| 1 | 600 | 35 |
| 2 | 600 | 36 |
| 3 | 600 | 17 |
| 4 | 600 | 22 |
| 5 | 600 | 26 |
| 6 | 600 | 28 |
| 7 | 600 | 21 |
| 8 | 600 | 20 |
| 9 | 600 | 18 |
| 10 | 600 | 28 |

**Verified — integrity:** no malformed records or missing endpoints were found. All 5,268 unique endpoint names resolve to JPG members under `03_Protocol_Images/` inside `03_Protocol_Images.tar`; all corresponding `.pts` members are also present. There are 231 unordered-pair groups repeated within folds, accounting for 251 excess pair rows. No unordered pair repeats across folds, and no repeated pair was found with endpoints reversed. The official pair rows should not be deduplicated or reordered.

**Verified — observed age differences:** the raw `annot=1` rows have an endpoint-age absolute difference of exactly 20 years (3,000 rows); raw `annot=-1` rows have a difference of exactly 5 years (3,000 rows). These are observations about the numeric labels and stored ages, not a stand-alone definition of label meaning. **Inferred:** the file’s “20_years” designation refers to the age gap of its same-identity pairs.

**Verified:** folds are explicit in the `splits` array and can be retained without reconstructing or repartitioning the protocol. A future run should keep every original pair row, fold number, endpoint order, and annotation unchanged.

## 3. Image decoding and landmark audit

**Verified:** the audit resolved all 5,268 distinct protocol JPG references directly from the tar stream, without extracting files. All **5,268/5,268** were decoded in memory using Windows `System.Drawing` (`Image.FromStream` with validation, drawing to a bitmap, and reading a pixel); there were **0 decoding failures** and no decoder exceptions.

**Verified:** all **5,268/5,268** matching `.pts` files were present and parseable. Each declared and contained exactly 68 finite coordinate pairs in a two-number-per-line coordinate block. No malformed landmark files or missing image/landmark pairs were found.

**Verified, flag for review:** **15** landmark files had at least one point outside the corresponding image’s `[0,width) x [0,height)` bounds. This is a coordinate-boundary check, not proof the landmark is wrong; inspect the affected records before using the 68-point annotations for alignment. No referenced JPG failed decoding. No image-quality or face-detection test was performed.

**Unresolved:** `.pts` availability does not establish that the original AgeDB benchmark used those landmarks. This audit did not establish the benchmark’s original alignment procedure.

**Read-only checks used:** an inline Python standard-library MAT v5 decoder (`struct`/`zlib`) enumerated protocol references and folds; a PowerShell tar-stream reader inspected matching members in memory; `System.Drawing` decoded JPGs; a text parser checked `.pts` headers, point counts, coordinate formatting, finite values, and image-bound extents. No helper files or extracted images were created.

## 4. Existing MORPH preprocessing pipeline

**Verified — code evidence:** `src\experiments\preprocess_morph.py` defines model pack name `buffalo_l` (line 30), BGR `uint8` OpenCV image loading/validation (lines 56–71), and ArcFace alignment to `112 x 112` with `norm_crop(..., image_size=OUTPUT_SIZE, mode="arcface")` (lines 77–99). Its `FaceAnalysis` is configured with `allowed_modules=["detection"]`, CPU execution, detector threshold `0.5`, and detection size `160 x 160` (lines 104–117). It requires exactly one detected face: zero faces, multiple faces, missing landmarks, or alignment errors are recorded as failures; it does not select a face when there are multiple detections.

**Verified — extraction code evidence:** `src\experiments\embedding_extraction.py` uses the same `buffalo_l` pack (lines 44, 169), loads a recognition model through `model_zoo.get_model` (lines 208–224, 252–256), and consumes the successful Stage 4 `processed_filepath` images (lines 138–156, 320–325). It validates `112 x 112 x 3 uint8` BGR inputs (lines 120–134), validates finite, nonzero 512-dimensional outputs, and L2-normalizes embeddings (lines 84–100, 243–245). Per-image extraction errors are caught and recorded (lines 311–337). The preprocessing code documents 0–255 BGR images and defers normalization to the model (preprocess lines 148–153, 389–392).

**Verified — version caveat:** `requirements.txt` pins `insightface==2.0` (lines 8–10), and the extraction code records a runtime package version in its manifest (embedding extraction line 465). The actual installed version used for prior MORPH results was not established by this source inspection.

**Unresolved:** code selects the `buffalo_l` pack but does not specify the resolved recognition ONNX filename or hash. The recognition model’s internal channel conversion/scaling is not explicit in the inspected extraction code. The exact comparison score and frozen MORPH threshold provenance/value were not verified within this audit’s scope. No separate blur or image-quality filter was found in the described preprocessing path.

## 5. AgeDB–MORPH compatibility matrix

| Dimension | MORPH pipeline (verified) | AgeDB 20-year files (verified) | Compatibility and open decision |
|---|---|---|---|
| Image source | MORPH images are loaded as BGR `uint8`, then successfully aligned images are consumed by Stage 5. | Original protocol images are JPG archive members; all referenced files decoded. Matching 68-point `.pts` files exist. | **Conditional:** run the MORPH preprocessing path on AgeDB source images to match handling, subject to approval and recorded failures. |
| Detection | `buffalo_l` detection module, CPU, threshold 0.5, detection input `160 x 160`; exactly one face required. | The archive contains images and landmarks; this audit did not run a detector. | **Unresolved until a separately approved preflight:** detector success, face count, and alignment status on AgeDB are unknown. |
| Alignment | Five-point detector landmarks passed to ArcFace `norm_crop`, output `112 x 112`. | 68-point `.pts` coordinates are parseable, but original benchmark use and a mapping to the MORPH five-point alignment are unverified. | **Conditional:** prefer the identical detector/alignment implementation; do not substitute `.pts` landmarks without a documented mapping and explicit approval. |
| Pixel format and scaling | BGR `uint8`, 0–255 at input; model-internal normalization is deferred. | JPG decoding is verified; the archive’s source images have not been run through the MORPH code. | **Unresolved:** verify the same channel order, value range, resize/crop, and model-internal preprocessing before inference. |
| Embedding model | Code selects `buffalo_l` and loads its recognition model via `model_zoo`; output is checked as 512-D and L2-normalized. | No AgeDB-specific embedding/model processing was done. | **Unresolved:** capture exact pack contents, ONNX filename/hash, package version, and score function used for MORPH. |
| Threshold and evaluation | Exact fixed MORPH threshold and its score/selection provenance were not checked here. | Pair protocol preserves folds and raw annotations. | **Conditional:** use only the already approved MORPH threshold on its original score scale; never choose or adjust a threshold on AgeDB. If provenance cannot be confirmed, do not report thresholded cross-dataset results. |
| Labels and identities | Not applicable. | Labels are numeric; identity-like first name tokens match for all `annot=1` pairs and differ for all `annot=-1` pairs. | **Inferred:** this pattern supports the common same/different interpretation but is not an author definition or proof of global identity-key uniqueness. |

## 6. Proposed leakage-safe evaluation protocol

1. Obtain approval for a separate external-evaluation run. Before extraction, record the exact recognition ONNX file and hash, InsightFace/runtime versions, preprocessing version, score function, and the frozen MORPH threshold with its MORPH-only selection provenance.
2. Resolve each protocol endpoint to its exact archive member. Preserve the original 10 folds, 600 pair rows per fold, endpoint order, and raw annotation. Do not deduplicate repeated rows or resample pairs.
3. Apply the exact MORPH detector/alignment and embedding pipeline to protocol images only if the compatibility decisions in Sections 4–5 are resolved. Keep all processing deterministic where possible. Do not use AgeDB to train, select, calibrate, or tune a model or threshold.
4. Evaluate AgeDB as its own external dataset. If approved, report metrics at the frozen MORPH threshold as external results; keep any threshold-free metrics distinct. Do not pool AgeDB pairs, scores, or metrics with MORPH.
5. Report each fold separately, with its original 600-row protocol denominator, scored-pair coverage, and failure counts. Summarize the ten fold results with the macro mean and standard deviation and report the protocol-wide coverage and aggregate separately. Any result on only successfully processed pairs must be labelled **evaluable-subset**, not a complete official-protocol result. Pair rows share images and include within-fold repeats, so do not treat all rows as independent observations for uncertainty intervals.
6. Keep identity-level AgeDB drift analysis on hold until the first filename token’s identity meaning, uniqueness scope, and stability across all observations are independently validated.

## 7. Failure handling and exclusion policy

- **Before extraction:** if a referenced image or landmark is missing, or an image cannot decode, stop the preflight and report the affected pair/fold. The current read-only audit found none of these cases.
- **Face detection/alignment:** use the MORPH policy unchanged. No face, multiple faces, missing required detector landmarks, or alignment error means the image is not eligible; do not choose a face manually, retry with a different detector, or silently substitute `.pts` landmarks.
- **Pair eligibility:** if either endpoint fails, retain the pair row and fold in the audit, mark it unscored with a reason, and report coverage against the original 600 rows. Do not replace the endpoint, remove the row from the protocol, or quietly score just the other endpoint. A valid-pair-only result is a separately labelled subset analysis; it must not be presented as full-protocol performance.
- **Landmarks:** flag the 15 out-of-frame cases for review if any future procedure proposes to use `.pts`. Do not use these points to repair detector/alignment failures without a separately specified and approved sensitivity analysis.
- **Protocol semantics:** retain raw `annot` values in all audit tables until label meaning is authorized. Do not reverse-map labels based solely on observed name-token agreement.

## 8. Planned output schemas

No outputs below were generated. Check AgeDB terms before storing or sharing them; pair-level identity/name tokens, annotations, and derived identity-linked data may be restricted.

| Future output | Proposed fields/schema | Meaning and redistribution caution |
|---|---|---|
| `agedb20_preprocessing_pair_audit.csv` | `protocol_file, fold, pair_index, raw_annot, endpoint_a_name, endpoint_b_name, endpoint_a_age, endpoint_b_age, image_member_a, image_member_b, jpg_present_a/b, decode_status_a/b, width_a/height_a, landmark_present_a/b, landmark_parse_status_a/b, point_count_a/b, out_of_bounds_a/b, alignment_status_a/b, extraction_status_a/b, exclusion_reason, pair_eligible` | Pair-level traceability and failure coverage; includes protocol annotations, ages, names, and derived statuses. Treat as potentially restricted; keep local unless license terms allow redistribution. |
| `agedb20_per_fold_results.csv` | `protocol_file, fold, n_protocol_pairs, n_scored_pairs, n_unscored_pairs, n_raw_label_1, n_raw_label_minus1, frozen_threshold_id, metric_name, metric_value, metric_definition` | Fold results and coverage at the pre-existing threshold; values are derived evaluation data, and redistribution rights are unresolved. |
| `agedb20_protocol_summary.json` or `.csv` | `protocol, model_pack, recognition_weight_filename, weight_hash, package_versions, preprocessing_version, score_function, threshold_value, threshold_source, fold_count, protocol_pair_count, scored_pair_count, fold_metric_mean, fold_metric_sd, exclusions_by_reason` | Reproducibility and protocol-level summary. Do not omit threshold provenance or exclusions. |
| `agedb20_vs_morph_external_comparison` figure | Dataset name, metric definition, frozen threshold identifier where applicable, fold summary, uncertainty/coverage annotation | Only produce if both datasets use compatible model, score, threshold, and metric definitions. Show AgeDB and MORPH separately; do not pool. Derived results may still be governed by dataset terms. |
| `agedb20_reproducibility.log` | Timestamp, code revision, package/runtime versions, model/weight identifiers and hashes, detector/alignment settings, input and output formats, threshold provenance, random seed if applicable, protocol/fold identifiers, counts, exclusions, failures | Audit trail without image pixels or copied landmarks. Avoid embedding licensed source material or personally identifying endpoint names. |

## 9. Licensing, limitations, and unresolved questions

- **Verified:** this audit did not copy images, landmarks, annotations, or derived outputs outside their existing archive/directory locations, and it generated no audit artifacts other than this requested report.
- **User-supplied licensing premise:** the AgeDB resource page is described in the request as permitting non-commercial research with redistribution restrictions. The current terms and whether protocol-derived CSVs, identity-linked records, embeddings, scores, or aggregate results may be redistributed were not independently verified. Check the current dataset terms and, if necessary, obtain permission before publishing or uploading any data-derived artifacts.
- **Unresolved protocol authority:** no author-hosted source defining the MAT `annot` values, identity-token scope, or exact sampling/fold rules was located in the sources checked. The MAT file itself only exposes `pairs` and `annot`.
- **Inferred label convention:** the public [InsightFace-derived AgeDB converter](https://github.com/bleakie/MaskInsightface/blob/94511404eaa7912945fa087e6445a3608c46aaea/src/data/agedb2pack.py#L100-L110) compares the first underscore-delimited name token and sets `issame=True` when the tokens match. That is useful corroboration for the token’s intended role in that conversion and matches this protocol’s observed raw labels; it is not author documentation or proof that the token is a globally unique and stable person ID.
- **Unresolved trajectory grouping:** pairwise protocol agreement does not establish that every image sharing a token depicts the same person, that different tokens always mean different people, or that all observations belong to one comparable longitudinal trajectory. Identity-level drift remains on hold pending authoritative identity semantics and validation.
- **Original paper citation:** Moschoglou et al., “AgeDB: The First Manually Collected, In-the-Wild Age Database,” *IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2017, pp. 1997–2005, DOI: [10.1109/CVPRW.2017.250](https://doi.org/10.1109/CVPRW.2017.250).

## 10. Approval checklist before any embedding extraction

- [ ] Confirm the current AgeDB license and permitted handling/publication of all planned derived artifacts.
- [ ] Confirm protocol annotation semantics and first-token identity scope from authoritative documentation or dataset-provider clarification.
- [ ] Record the resolved `buffalo_l` recognition weight filename and cryptographic hash; verify the actual InsightFace/runtime version.
- [ ] Confirm the exact MORPH score function and frozen threshold value, along with its MORPH-only selection provenance; do not estimate or tune it on AgeDB.
- [ ] Approve the image path from archive member to MORPH-compatible BGR `uint8` input, detector settings, five-point alignment, and 112-pixel output. Decide explicitly whether `.pts` landmarks are unused; if proposed for alignment, resolve the 15 out-of-frame cases and validate the mapping first.
- [ ] Approve the failure policy and output schemas, including preservation of all original folds and explicit coverage/exclusion counts.
- [ ] Authorize a separate run. Until then, do not extract embeddings or calculate any evaluation result.

**No-evaluation confirmation:** this task performed only read-only protocol, archive-image decoding, landmark parsing, and source/configuration inspection. No model inference, embeddings, similarity scores, threshold selection/validation, or evaluation metrics were computed.
