#!/usr/bin/env bash
set -euo pipefail

#
# Smoke tests for a deployed orchestrator-rhoai system.
# Run after 'make deploy-tier{1,2,3}'.
#

NAMESPACE="${NAMESPACE:-orchestrator-rhoai}"
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }

echo "============================================="
echo " orchestrator-rhoai — Smoke Tests"
echo " Namespace: $NAMESPACE"
echo "============================================="
echo ""

# ── 1. Namespace exists ─────────────────────────────────────────────────────

echo "1. Namespace"
if oc get namespace "$NAMESPACE" &>/dev/null; then
    pass "Namespace '$NAMESPACE' exists"
else
    fail "Namespace '$NAMESPACE' not found"
    echo "Cannot continue. Deploy first with 'make deploy-tier1'."
    exit 1
fi
echo ""

# ── 2. InferenceServices ────────────────────────────────────────────────────

echo "2. InferenceServices"
ISVC_LIST=$(oc get inferenceservice -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' 2>/dev/null)

if [ -z "$ISVC_LIST" ]; then
    fail "No InferenceServices found"
else
    while IFS= read -r line; do
        NAME=$(echo "$line" | awk '{print $1}')
        READY=$(echo "$line" | awk '{print $2}')
        if [ "$READY" = "True" ]; then
            pass "InferenceService '$NAME' is Ready"
        else
            fail "InferenceService '$NAME' is NOT Ready (status: $READY)"
        fi
    done <<< "$ISVC_LIST"
fi
echo ""

# ── 3. Gateway Deployment ───────────────────────────────────────────────────

echo "3. Gateway Service"
if oc get deployment orchestrator-gateway -n "$NAMESPACE" &>/dev/null; then
    READY_REPLICAS=$(oc get deployment orchestrator-gateway -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "${READY_REPLICAS:-0}" -gt 0 ]; then
        pass "Gateway deployment has $READY_REPLICAS ready replica(s)"
    else
        fail "Gateway deployment has 0 ready replicas"
    fi
else
    fail "Gateway deployment not found"
fi
echo ""

# ── 4. Gateway Health Endpoint ──────────────────────────────────────────────

echo "4. Gateway Health Check"
GATEWAY_URL=$(oc get route orchestrator-gateway -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

if [ -n "$GATEWAY_URL" ]; then
    HTTP_CODE=$(curl -sk -o /dev/null -w '%{http_code}' "https://${GATEWAY_URL}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        pass "Gateway /health returns 200"
        HEALTH_BODY=$(curl -sk "https://${GATEWAY_URL}/health" 2>/dev/null)
        echo "       Response: $HEALTH_BODY"
    else
        fail "Gateway /health returned HTTP $HTTP_CODE"
    fi
else
    fail "Gateway Route not found — cannot test health endpoint"
fi
echo ""

# ── 5. Gateway Readiness ───────────────────────────────────────────────────

echo "5. Gateway Readiness"
if [ -n "$GATEWAY_URL" ]; then
    HTTP_CODE=$(curl -sk -o /dev/null -w '%{http_code}' "https://${GATEWAY_URL}/ready" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        pass "Gateway /ready returns 200"
    else
        fail "Gateway /ready returned HTTP $HTTP_CODE"
    fi
else
    fail "Gateway Route not found"
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────────────────

echo "============================================="
echo " Smoke Test Results: $PASS passed, $FAIL failed"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Some smoke tests failed. Check the deployment."
    exit 1
else
    echo ""
    echo "All smoke tests passed."
    exit 0
fi
