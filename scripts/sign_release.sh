#!/usr/bin/env bash
# Sign the release artifacts and the supply-chain evidence out of band.
#
# This script is run by a release owner on a trusted machine, never in CI, so no signing key ever
# enters the build. It signs the built distributions in dist plus the committed bill of materials and
# provenance, producing a detached signature next to each file. The committed provenance already binds
# the version to the exact dependency tree; signing binds those, and the built artifacts, to the
# owner's key.
#
# The signer is configurable. Set REFAAT_SIGN_CMD to a command that takes one file path and writes a
# detached signature; the default uses gpg. Examples:
#   export REFAAT_SIGN_CMD='gpg --armor --detach-sign --output "$1.asc" "$1"'
#   export REFAAT_SIGN_CMD='cosign sign-blob --yes --output-signature "$1.sig" "$1"'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIGN_CMD="${REFAAT_SIGN_CMD:-gpg --armor --detach-sign --output \"\$1.asc\" \"\$1\"}"

targets=()
while IFS= read -r -d '' path; do targets+=("$path"); done < <(find "$ROOT/dist" -maxdepth 1 -type f -print0 2>/dev/null || true)
targets+=("$ROOT/docs/reference/sbom.json" "$ROOT/docs/reference/provenance.json")

if [ "${#targets[@]}" -eq 2 ]; then
  echo "sign_release: no built artifacts in dist; run 'python -m build' first" >&2
  exit 1
fi

sign_one() { bash -c "$SIGN_CMD" _ "$1"; }

for target in "${targets[@]}"; do
  if [ ! -f "$target" ]; then
    echo "sign_release: missing $target" >&2
    exit 1
  fi
  echo "sign_release: signing $target"
  sign_one "$target"
done

echo "sign_release: signed ${#targets[@]} files; publish the signatures alongside the artifacts"
