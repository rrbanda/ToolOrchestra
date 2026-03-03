#!/usr/bin/env bash
set -euo pipefail

#
# Verify that an OpenShift cluster meets the prerequisites for
# orchestrator-rhoai deployment.
#

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN + 1)); }

echo "============================================="
echo " orchestrator-rhoai — Cluster Readiness Check"
echo "============================================="
echo ""

# ── 1. oc CLI connectivity ──────────────────────────────────────────────────

echo "1. OpenShift Connectivity"
if oc whoami &>/dev/null; then
    USER=$(oc whoami)
    SERVER=$(oc whoami --show-server)
    pass "Logged in as '$USER' to $SERVER"
else
    fail "Not logged in. Run 'oc login' first."
    echo ""
    echo "Cannot continue without cluster access. Exiting."
    exit 1
fi

VERSION=$(oc get clusterversion version -o jsonpath='{.status.desired.version}' 2>/dev/null || echo "unknown")
echo "       Cluster version: $VERSION"
echo ""

# ── 2. Required Operators ───────────────────────────────────────────────────

echo "2. Required Operators"

check_operator() {
    local name="$1"
    local namespace="$2"
    local pattern="$3"
    local result
    result=$(oc get csv -n "$namespace" --no-headers 2>/dev/null | grep "$pattern" | head -1 || true)
    if [ -n "$result" ]; then
        local ver
        ver=$(echo "$result" | awk '{print $NF}')
        pass "$name ($ver)"
    else
        fail "$name (not found in $namespace)"
    fi
}

check_operator "Red Hat OpenShift AI"    "redhat-ods-operator"       "rhods-operator"
check_operator "NVIDIA GPU Operator"     "nvidia-gpu-operator"       "gpu-operator-certified"
check_operator "Node Feature Discovery"  "openshift-nfd"             "nfd"
check_operator "OpenShift Serverless"    "openshift-serverless"      "serverless-operator"
check_operator "OpenShift Service Mesh"  "default"                   "servicemeshoperator"
check_operator "OpenShift Pipelines"     "openshift-pipelines"       "openshift-pipelines"
echo ""

# ── 3. RHOAI Components ────────────────────────────────────────────────────

echo "3. RHOAI DataScienceCluster Components"

check_component() {
    local component="$1"
    local state
    state=$(oc get datasciencecluster -A -o jsonpath="{.items[0].spec.components.${component}.managementState}" 2>/dev/null || echo "not-found")
    if [ "$state" = "Managed" ]; then
        pass "$component: Managed"
    elif [ "$state" = "Removed" ]; then
        warn "$component: Removed (may be needed later)"
    else
        fail "$component: $state"
    fi
}

check_component "kserve"
check_component "ray"
check_component "trainingoperator"
check_component "dashboard"
check_component "workbenches"
check_component "modelregistry"
check_component "aipipelines"
echo ""

# ── 4. GPU Nodes ────────────────────────────────────────────────────────────

echo "4. GPU Availability"

GPU_NODES=$(oc get nodes -o json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
gpu_nodes = []
for node in data.get('items', []):
    gpu = node.get('status', {}).get('capacity', {}).get('nvidia.com/gpu', '0')
    if int(gpu) > 0:
        name = node['metadata']['name']
        labels = node['metadata'].get('labels', {})
        gpu_type = labels.get('nvidia.com/gpu.product', 'unknown')
        gpu_mem = labels.get('nvidia.com/gpu.memory', 'unknown')
        gpu_nodes.append(f'{name}: {gpu}x {gpu_type} ({gpu_mem} MB)')
for n in gpu_nodes:
    print(n)
" 2>/dev/null)

if [ -n "$GPU_NODES" ]; then
    pass "GPU nodes found:"
    echo "$GPU_NODES" | while read -r line; do
        echo "       $line"
    done
else
    fail "No GPU nodes detected"
fi
echo ""

# ── 5. Storage Classes ──────────────────────────────────────────────────────

echo "5. Storage Classes"

if oc get sc gp3-csi &>/dev/null; then
    pass "gp3-csi storage class available"
elif oc get sc gp2-csi &>/dev/null; then
    warn "gp2-csi available but gp3-csi preferred. Update PVC manifests if needed."
else
    warn "Neither gp3-csi nor gp2-csi found. Check available storage classes."
fi
echo ""

# ── 6. KServe Readiness ────────────────────────────────────────────────────

echo "6. KServe / Model Serving"

if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
    pass "InferenceService CRD exists"
else
    fail "InferenceService CRD not found (KServe not installed)"
fi

if oc get crd servingruntimes.serving.kserve.io &>/dev/null; then
    pass "ServingRuntime CRD exists"
else
    fail "ServingRuntime CRD not found"
fi
echo ""

# ── 7. KubeRay Readiness ───────────────────────────────────────────────────

echo "7. KubeRay (for training pipeline)"

if oc get crd rayclusters.ray.io &>/dev/null; then
    pass "RayCluster CRD exists"
else
    warn "RayCluster CRD not found (needed for Phase 6 training)"
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────────────────

echo "============================================="
echo " Summary: $PASS passed, $FAIL failed, $WARN warnings"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Some checks failed. Please resolve before deploying."
    exit 1
else
    echo ""
    echo "Cluster is ready for orchestrator-rhoai deployment."
    exit 0
fi
