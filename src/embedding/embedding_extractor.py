#src/embedding/embedding_extractor.py

"""ArcFace embedding extraction with InsightFace.

This module provides the first reusable embedding extraction component for the
project. It uses InsightFace's pretrained FaceAnalysis pipeline and does not
train, fine-tune, or modify any recognition model.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from insightface import model_zoo
from insightface.app import FaceAnalysis
from insightface.app.common import Face


LOGGER = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []

FaceSelectionMode = Literal["error", "largest", "highest_confidence"]


class FaceEmbeddingError(Exception):
    """Base exception for face embedding extraction errors."""


class ImageReadError(FaceEmbeddingError):
    """Raised when an image path does not exist or cannot be read."""


class ExtractorNotInitializedError(FaceEmbeddingError):
    """Raised when extraction is requested before the model is initialized."""


class NoFaceDetectedError(FaceEmbeddingError):
    """Raised when InsightFace does not detect any face in an image."""


class MultipleFacesDetectedError(FaceEmbeddingError):
    """Raised when multiple faces are found and no explicit selection is allowed."""


class FaceEmbeddingExtractor:
    """Extract ArcFace face embeddings with InsightFace.

    Parameters
    ----------
    model_name:
        InsightFace model pack name. ``buffalo_l`` is the default ArcFace-based
        pack commonly used with ``FaceAnalysis``.
    model_root:
        Optional InsightFace cache directory. Leave as ``None`` to use the
        package default outside this source tree.
    det_size:
        Detector input size passed to ``FaceAnalysis.prepare``.
    det_thresh:
        Face detection threshold.
    normalize:
        Whether to return L2-normalized embeddings. ArcFace comparisons usually
        use normalized embeddings for cosine similarity.
    face_selection:
        How to handle multiple detected faces when one embedding is requested.
        ``error`` raises, ``largest`` chooses the largest bounding box, and
        ``highest_confidence`` chooses the highest detector score.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        normalize: bool = True,
        face_selection: FaceSelectionMode = "error",
    ) -> None:
        self.model_name = model_name
        self.model_root = Path(model_root).expanduser() if model_root else None
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.normalize = normalize
        self.face_selection = face_selection

        self.app: FaceAnalysis | None = None
        self.providers: list[str] = []
        self.active_providers: list[str] = []
        self.embedding_dimension: int | None = None
        self._aligned_recognition_model: object | None = None

        if face_selection not in ("error", "largest", "highest_confidence"):
            raise ValueError(
                "face_selection must be one of: error, largest, highest_confidence"
            )

    @staticmethod
    def available_providers() -> list[str]:
        """Return the ONNX Runtime execution providers available locally."""

        return list(ort.get_available_providers())

    @staticmethod
    def preferred_providers() -> list[str]:
        """Choose CUDA when available, with CPU as a fallback provider."""

        available = FaceEmbeddingExtractor.available_providers()
        providers: list[str] = []

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        if not providers:
            raise RuntimeError(
                "No supported ONNX Runtime provider is available. Expected "
                "CUDAExecutionProvider or CPUExecutionProvider."
            )

        return providers

    def initialize(self) -> None:
        """Initialize InsightFace's pretrained detection and recognition models.

        The first call may download the selected InsightFace model pack into the
        InsightFace cache directory if it is not already present.
        """

        self.providers = self.preferred_providers()
        ctx_id = 0 if self.providers[0] == "CUDAExecutionProvider" else -1

        LOGGER.info("Initializing InsightFace model pack '%s'.", self.model_name)
        LOGGER.info("Requested ONNX Runtime providers: %s", self.providers)

        self._preload_onnxruntime_dlls()

        try:
            self._initialize_with_providers(self.providers, ctx_id=ctx_id)
        except Exception:
            if self.providers == ["CPUExecutionProvider"]:
                raise

            LOGGER.exception(
                "InsightFace initialization failed with CUDA. Retrying with CPU only."
            )
            self.app = None
            self.active_providers = []
            self.providers = ["CPUExecutionProvider"]
            self._initialize_with_providers(self.providers, ctx_id=-1)

        self.active_providers = self._active_model_providers()
        if self.active_providers:
            LOGGER.info("Active ONNX Runtime providers: %s", self.active_providers)

        LOGGER.info("InsightFace initialized successfully.")
        self._aligned_recognition_model = None

    def _initialize_with_providers(self, providers: list[str], ctx_id: int) -> None:
        """Create and prepare the InsightFace app with a provider list."""

        kwargs: dict[str, object] = {
            "name": self.model_name,
            "allowed_modules": ["detection", "recognition"],
            "providers": providers,
        }
        if self.model_root is not None:
            kwargs["root"] = str(self.model_root)

        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh, det_size=self.det_size)

    @staticmethod
    def _preload_onnxruntime_dlls() -> None:
        """Preload CUDA/cuDNN DLLs from Python packages before ORT sessions start."""

        for dll_dir in FaceEmbeddingExtractor._nvidia_dll_directories():
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            except (AttributeError, OSError):
                LOGGER.debug("Could not register DLL directory: %s", dll_dir)

        FaceEmbeddingExtractor._prepend_process_path(
            FaceEmbeddingExtractor._nvidia_dll_directories()
        )

        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is None:
            return

        preload_dlls(directory="")

    @staticmethod
    def _nvidia_dll_directories() -> list[Path]:
        """Return NVIDIA Python package DLL directories for this environment."""

        site_packages = Path(ort.__file__).resolve().parents[1]
        nvidia_root = site_packages / "nvidia"
        candidates = [
            nvidia_root / "cuda_runtime" / "bin",
            nvidia_root / "cublas" / "bin",
            nvidia_root / "cudnn" / "bin",
            nvidia_root / "cuda_nvrtc" / "bin",
            nvidia_root / "cufft" / "bin",
            nvidia_root / "curand" / "bin",
            nvidia_root / "nvjitlink" / "bin",
        ]
        return [path for path in candidates if path.exists()]

    @staticmethod
    def _prepend_process_path(paths: list[Path]) -> None:
        """Expose DLL directories to libraries that search the process PATH."""

        current_paths = os.environ.get("PATH", "").split(os.pathsep)
        new_paths = [str(path) for path in paths if str(path) not in current_paths]
        if new_paths:
            os.environ["PATH"] = os.pathsep.join(new_paths + current_paths)

    def extract_embedding(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk and return one face embedding.

        Parameters
        ----------
        image_path:
            Path to an image readable by OpenCV.

        Returns
        -------
        numpy.ndarray
            A ``float32`` ArcFace embedding. The dimensionality is recorded in
            ``self.embedding_dimension`` after extraction.

        Raises
        ------
        ImageReadError
            If the image does not exist or OpenCV cannot read it.
        NoFaceDetectedError
            If no face is detected.
        MultipleFacesDetectedError
            If multiple faces are detected while ``face_selection='error'``.
        """

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise ImageReadError(f"Image file does not exist: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ImageReadError(f"OpenCV could not read image file: {path}")

        LOGGER.info("Extracting face embedding from image: %s", path)
        return self.extract_from_image(image)

    def extract_aligned_embedding(self, image: np.ndarray) -> np.ndarray:
        """Return an ArcFace embedding for an already-aligned face crop.

        This method is specifically for validation images that are already tightly
        cropped and resized to 112x112. It bypasses the detector and calls the
        ArcFace recognition model directly on the aligned crop, which avoids the
        no-face detection failure that occurs when the input is a crop rather than
        a full photograph.
        """

        if not isinstance(image, np.ndarray):
            raise ImageReadError("Image must be a NumPy array.")
        if image.size == 0:
            raise ImageReadError("Image array is empty.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        model = self._aligned_recognition_model
        if model is None:
            providers = self.providers or self.preferred_providers()
            self.providers = providers
            if providers[0] == "CUDAExecutionProvider":
                ctx_id = 0
            else:
                ctx_id = -1
            model = model_zoo.get_model(
                self.model_name,
                providers=providers,
                root=str(self.model_root) if self.model_root is not None else "~/.insightface",
            )
            if model is None:
                raise FaceEmbeddingError(
                    "InsightFace did not provide a recognition model for the selected model name."
                )
            if hasattr(model, "prepare"):
                model.prepare(ctx_id=ctx_id)
            self._aligned_recognition_model = model

        if not hasattr(model, "get_feat"):
            raise FaceEmbeddingError(
                "Aligned-face recognition model must expose the get_feat() API."
            )

        feature = np.asarray(model.get_feat(image), dtype=np.float32)
        embedding = feature.reshape(-1).astype(np.float32, copy=False)
        if self.normalize:
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                raise FaceEmbeddingError("Aligned-face embedding has zero norm.")
            embedding = (embedding / norm).astype(np.float32, copy=False)

        self.embedding_dimension = int(embedding.shape[0])
        return embedding

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Return one embedding from an OpenCV BGR image array."""

        faces = self._detect_faces(image)

        if not faces:
            raise NoFaceDetectedError("No face was detected in the image.")

        if len(faces) > 1:
            if self.face_selection == "error":
                raise MultipleFacesDetectedError(
                    f"Detected {len(faces)} faces. Set face_selection to "
                    "'largest' or 'highest_confidence' to choose explicitly."
                )
            face = self._select_face(faces)
            LOGGER.info(
                "Detected %d faces; selected one using '%s'.",
                len(faces),
                self.face_selection,
            )
        else:
            face = faces[0]

        return self._embedding_from_face(face)

    def extract_all_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Return embeddings for every detected face in an OpenCV BGR image array."""

        faces = self._detect_faces(image)
        if not faces:
            raise NoFaceDetectedError("No face was detected in the image.")

        return [self._embedding_from_face(face) for face in faces]

    def _detect_faces(self, image: np.ndarray) -> list[Face]:
        """Run InsightFace detection and recognition for a BGR image."""

        if self.app is None:
            raise ExtractorNotInitializedError(
                "FaceEmbeddingExtractor.initialize() must be called before extraction."
            )

        if not isinstance(image, np.ndarray):
            raise ImageReadError("Image must be a NumPy array.")
        if image.size == 0:
            raise ImageReadError("Image array is empty.")

        faces = self.app.get(image)
        return list(faces)

    def _embedding_from_face(self, face: Face) -> np.ndarray:
        """Read and validate a face embedding from an InsightFace Face object."""

        embedding = face.normed_embedding if self.normalize else face.embedding
        if embedding is None:
            raise FaceEmbeddingError(
                "InsightFace did not return a recognition embedding for the detected face."
            )

        embedding_array = np.asarray(embedding, dtype=np.float32)
        self.embedding_dimension = int(embedding_array.shape[0])
        return embedding_array

    def _select_face(self, faces: list[Face]) -> Face:
        """Select one face according to the configured multiple-face policy."""

        if self.face_selection == "largest":
            return max(faces, key=self._face_area)
        if self.face_selection == "highest_confidence":
            return max(faces, key=lambda face: float(face.det_score or 0.0))
        raise MultipleFacesDetectedError(
            f"Detected {len(faces)} faces, but face_selection is set to 'error'."
        )

    def _active_model_providers(self) -> list[str]:
        """Return providers reported by the first available ONNX model session."""

        if self.app is None:
            return []

        for model in self.app.models.values():
            session = getattr(model, "session", None)
            if session is not None and hasattr(session, "get_providers"):
                return list(session.get_providers())

        return []

    @staticmethod
    def _face_area(face: Face) -> float:
        """Compute bounding-box area for an InsightFace Face object."""

        bbox = np.asarray(face.bbox, dtype=np.float32)
        width = max(float(bbox[2] - bbox[0]), 0.0)
        height = max(float(bbox[3] - bbox[1]), 0.0)
        return width * height
