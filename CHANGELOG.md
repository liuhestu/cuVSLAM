# Changelog

## [15.0.0] - 2026-03-02

Initial open-source release.

### Added
- Cached map-to-disk SLAM mode
- RGBD pipeline optimization
- Rerun visualization of internal cuVSLAM data
- Examples moved into the repository (from a previously separate pycuvslam repo)
- Troubleshooting guide (`TROUBLESHOOTING.md`)

### Changed
- New NVIDIA Community License
- Switched to semantic versioning (from this release onward)
- SLAM internal refactoring is in progress
- Removed JPEG/PNG dependencies from libcuvslam
- Opened cuVSLAM API to accept user-provided `cudaStream_t`
- CMake 4+ build compatibility

### Removed
- C API (superseded by C++ API)
- Several internal methods and unused parameters from public SLAM API

### Fixed
- Memory leaks
- RGBD depth mask bug
- Multiple other bugfixes

### Security
- Updated libpng version
