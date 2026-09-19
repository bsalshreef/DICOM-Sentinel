# DICOM-Sentinel

DICOM-Sentinel is an **auditing framework, not a de-identification tool**. It evaluates privacy transformation, residual metadata linkability, DICOM structural integrity, controlled tampering sensitivity, pixel preservation, analytical utility, and external paired validation.

## Datasets

The development dataset is the [TCIA Pseudo-PHI-DICOM dataset](https://www.cancerimagingarchive.net/collection/pseudo-phi-dicom-data/). External paired validation uses the [MIDI-B Validation dataset](https://www.cancerimagingarchive.net/collection/midi-b-test-midi-b-validation/). Source DICOM files are not included in this repository. Data must be obtained directly from TCIA and used under the applicable data-use terms.

Official Patient-ID and UID mappings are not redistributed here unless their license permits redistribution. Obtain authorized mappings from the corresponding official data source.

## Analysis modules

- **E1:** baseline PHI audit
- **E2:** residual privacy leakage screening
- **E3/value-aware:** paired value-aware privacy analysis (`e2_value_aware_paired_v3.py`)
- **E4:** metadata linkability analysis
- **E5:** structural integrity analysis
- **E6:** reference-aware controlled-tampering analysis
- **E7:** pixel and analytical-utility analysis
- **E7b:** anomaly characterization
- **E8a:** external paired validation on MIDI-B Validation

The scripts are frozen analysis artifacts. Their logic is not modified by this repository packaging.

## Environment

`requirements_lock.txt` records the complete Python environment used for the final analyses. `environment_lock.txt` records the Python, operating-system, and key scientific-package versions.

## License

The repository documentation and original project material are distributed under the Apache License 2.0. Third-party datasets, mappings, and dependencies remain subject to their own licenses and terms.
