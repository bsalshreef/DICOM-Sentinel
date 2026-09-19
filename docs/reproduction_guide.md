# Reproduction Guide

Obtain the authorized Pseudo-PHI-DICOM and MIDI-B Validation data from TCIA. Place the paired DICOM folders and any authorized mapping files outside this repository. Do not commit DICOM files or restricted mappings.

Install the exact environment recorded in `requirements_lock.txt`, or use an equivalent Python environment matching `environment_lock.txt`. Run each script with its command-line help first, then supply the input and output paths required by the script. Preserve the supplied scripts and source DICOM files unchanged.

The E1 and E2 scripts operate on the Pseudo-PHI-DICOM evaluation and de-identified folders. The E3/value-aware, E4, E5, E6, and E7/E7b modules use the corresponding internal paired inputs and mappings. E8a uses the complete MIDI-B Synthetic Validation and Curated Validation folders together with the official Validation UID and Patient-ID mappings.

Results are written to user-selected output directories and are not included in this repository.
