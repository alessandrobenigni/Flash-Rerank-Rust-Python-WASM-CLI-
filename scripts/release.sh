#!/usr/bin/env bash
# release.sh — Build and package flash-rerank for distribution.
#
# Creates a self-contained tarball/zip containing the flash-rerank binary
# and a lib/ directory with bundled shared libraries (ONNX Runtime, etc.).
#
# Usage:
#   ./scripts/release.sh [--target <triple>]
#
# Examples:
#   ./scripts/release.sh
#   ./scripts/release.sh --target x86_64-unknown-linux-gnu

set -euo pipefail

VERSION=$(cargo metadata --format-version=1 --no-deps | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)
TARGET="${2:-$(rustc -vV | grep host | awk '{print $2}')}"
DIST_NAME="flash-rerank-${VERSION}-${TARGET}"
DIST_DIR="dist/${DIST_NAME}"

echo "=== Flash-Rerank Release Build ==="
echo "Version:  ${VERSION}"
echo "Target:   ${TARGET}"
echo "Output:   ${DIST_DIR}"
echo ""

# 1. Build release binary
echo "[1/4] Building release binary..."
cargo build --release --package flash-rerank-cli --target "${TARGET}" 2>&1

# 2. Create distribution directory
echo "[2/4] Creating distribution layout..."
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/lib"

# 3. Copy binary
BINARY_NAME="flash-rerank"
if [[ "${TARGET}" == *"windows"* ]]; then
    BINARY_NAME="flash-rerank.exe"
fi
cp "target/${TARGET}/release/${BINARY_NAME}" "${DIST_DIR}/"

# 4. Bundle shared libraries
echo "[3/4] Bundling shared libraries..."

# Attempt to locate and copy ONNX Runtime shared library
if [[ "${TARGET}" == *"linux"* ]]; then
    # Look for libonnxruntime in common locations
    for lib_path in \
        "${ORT_LIB_LOCATION:-}" \
        "/usr/local/lib" \
        "/usr/lib" \
        "${HOME}/.onnxruntime/lib"; do
        if [[ -n "${lib_path}" && -f "${lib_path}/libonnxruntime.so" ]]; then
            cp "${lib_path}"/libonnxruntime*.so* "${DIST_DIR}/lib/" 2>/dev/null || true
            echo "  Bundled ONNX Runtime from ${lib_path}"
            break
        fi
    done
elif [[ "${TARGET}" == *"darwin"* ]]; then
    for lib_path in \
        "${ORT_LIB_LOCATION:-}" \
        "/usr/local/lib" \
        "${HOME}/.onnxruntime/lib"; do
        if [[ -n "${lib_path}" && -f "${lib_path}/libonnxruntime.dylib" ]]; then
            cp "${lib_path}"/libonnxruntime*.dylib "${DIST_DIR}/lib/" 2>/dev/null || true
            echo "  Bundled ONNX Runtime from ${lib_path}"
            break
        fi
    done
elif [[ "${TARGET}" == *"windows"* ]]; then
    for lib_path in \
        "${ORT_LIB_LOCATION:-}" \
        "C:/Program Files/onnxruntime/lib"; do
        if [[ -n "${lib_path}" && -f "${lib_path}/onnxruntime.dll" ]]; then
            cp "${lib_path}"/onnxruntime*.dll "${DIST_DIR}/lib/" 2>/dev/null || true
            echo "  Bundled ONNX Runtime from ${lib_path}"
            break
        fi
    done
fi

# 5. Package
echo "[4/4] Packaging..."
cd dist
if [[ "${TARGET}" == *"windows"* ]]; then
    zip -r "${DIST_NAME}.zip" "${DIST_NAME}/"
    echo ""
    echo "Package: dist/${DIST_NAME}.zip"
else
    tar czf "${DIST_NAME}.tar.gz" "${DIST_NAME}/"
    echo ""
    echo "Package: dist/${DIST_NAME}.tar.gz"
fi

echo ""
echo "=== Release build complete ==="
echo "Contents:"
ls -la "${DIST_NAME}/"
ls -la "${DIST_NAME}/lib/" 2>/dev/null || echo "  (lib/ is empty — add shared libraries manually if needed)"
