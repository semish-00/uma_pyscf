"""Versioned, typed record schemas and their validators.

``schemas`` sits directly above ``core`` in the one-way dependency chain
``core -> schemas -> (sampling | calculators | qc | datasets | ...) -> cli``.
It owns what a record *is* -- the field set, the units, the invariants, and the
``schema`` string that versions all of it -- and owns none of the science that
fills one in. The fail-closed checks the project requires (charge/spin parity,
units, required fields) live here at the library's entrance instead of being
repeated by every module that later handles a record.

Modules exchange records, not imports: a producer writes
:meth:`LabelRecord.to_dict` through ``core.io.write_json_atomic`` and a consumer
reads it back through :meth:`LabelRecord.from_dict`.
"""

from __future__ import annotations

from .acquisition import (
    ACQUISITION_SCORE_SCHEMA,
    SELECTION_MANIFEST_SCHEMA,
    AcquisitionScoreManifest,
    AcquisitionScoreRecord,
    SelectionManifest,
)
from .candidate import (
    CANDIDATE_MANIFEST_SCHEMA,
    CANDIDATE_STATUSES,
    GEOMETRY_QC_SCHEMA,
    CandidateManifest,
    CandidateRecord,
    GeometryQcReport,
)
from .cli import configure_validate_records, run_validate_records
from .composition_baseline import (
    COMPOSITION_BASELINE_METHOD,
    COMPOSITION_BASELINE_SCHEMA,
    CompositionBaseline,
)
from .crosscode import (
    CROSSCODE_RESULT_SCHEMA,
    IMPORT_EVENT,
    label_record_from_crosscode_result,
)
from .dataset_manifest import ASE_DATASET_MANIFEST_SCHEMA, AseDatasetManifest
from .label_record import (
    CANONICAL_UNITS,
    LABEL_RECORD_SCHEMA,
    QC_STATUSES,
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)
from .model_prediction import (
    MODEL_PREDICTION_SCHEMA,
    PREDICTION_UNITS,
    ModelPredictionManifest,
    ModelPredictionRecord,
)
from .qc_report import (
    QC_REPORT_SCHEMA,
    QC_REPORT_STATUSES,
    QcReport,
)
from .split_manifest import (
    FRACTION_SUM_TOLERANCE,
    SPLIT_AXES,
    SPLIT_MANIFEST_SCHEMA,
    SplitManifest,
    validate_axis,
    validate_partitions,
)
from .state_registry import (
    STATE_REGISTRY_SCHEMA,
    STATE_REGISTRY_STATUSES,
    StateRegistry,
    StateRegistryEntry,
)

__all__ = [
    "ACQUISITION_SCORE_SCHEMA",
    "ASE_DATASET_MANIFEST_SCHEMA",
    "CANDIDATE_MANIFEST_SCHEMA",
    "CANDIDATE_STATUSES",
    "COMPOSITION_BASELINE_METHOD",
    "COMPOSITION_BASELINE_SCHEMA",
    "CANONICAL_UNITS",
    "CROSSCODE_RESULT_SCHEMA",
    "FRACTION_SUM_TOLERANCE",
    "GEOMETRY_QC_SCHEMA",
    "IMPORT_EVENT",
    "LABEL_RECORD_SCHEMA",
    "MODEL_PREDICTION_SCHEMA",
    "PREDICTION_UNITS",
    "QC_REPORT_SCHEMA",
    "QC_REPORT_STATUSES",
    "QC_STATUSES",
    "SPLIT_AXES",
    "SPLIT_MANIFEST_SCHEMA",
    "SELECTION_MANIFEST_SCHEMA",
    "STATE_REGISTRY_SCHEMA",
    "STATE_REGISTRY_STATUSES",
    "CandidateManifest",
    "CandidateRecord",
    "AcquisitionScoreManifest",
    "AcquisitionScoreRecord",
    "AseDatasetManifest",
    "CompositionBaseline",
    "ElectronicState",
    "Engine",
    "GeometryQcReport",
    "LabelRecord",
    "Method",
    "ModelPredictionManifest",
    "ModelPredictionRecord",
    "QcReport",
    "QcState",
    "RawArtifact",
    "Results",
    "SelectionManifest",
    "SplitManifest",
    "StateRegistry",
    "StateRegistryEntry",
    "Structure",
    "configure_validate_records",
    "label_record_from_crosscode_result",
    "run_validate_records",
    "validate_axis",
    "validate_partitions",
]
