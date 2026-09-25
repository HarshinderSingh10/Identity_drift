# Embedding Module

Face embeddings are compact numeric representations of face images. Instead of
comparing raw pixels, a face recognition model maps each detected and aligned
face into a vector space where images of the same identity should be close and
images of different identities should be farther apart.

In this project, ArcFace is used through InsightFace's pretrained
`FaceAnalysis` pipeline. ArcFace is not trained from scratch here. It provides
the recognition embedding that later stages will use for drift analysis,
stability scoring, and re-enrollment decisions.

Embeddings are central to identity drift detection because drift is measured as
change in this vector representation across time, pose, lighting, expression,
image quality, and other acquisition conditions. Once embeddings are extracted,
future modules can compare same-identity and different-identity vectors without
depending on the original image pixels.

The selected InsightFace recognition model is expected to return 512-dimensional
embeddings, but project code should verify this programmatically from the
returned NumPy array shape. The extractor records the observed dimensionality in
`FaceEmbeddingExtractor.embedding_dimension` after a successful extraction.

The initial extractor assumes dataset images usually contain one primary face.
If multiple faces are detected, the default behavior is to raise a clear error.
Callers can explicitly configure `face_selection="largest"` or
`face_selection="highest_confidence"` when that behavior is desired.

