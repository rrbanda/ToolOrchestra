#!/usr/bin/env bash
#
# Example API calls against a deployed orchestrator-rhoai system.
#

GATEWAY_URL="${GATEWAY_URL:-https://orchestrator-gateway-orchestrator-rhoai.apps.example.com}"

echo "=== Health Check ==="
curl -sk "${GATEWAY_URL}/health" | python3 -m json.tool
echo ""

echo "=== Readiness Check ==="
curl -sk "${GATEWAY_URL}/ready" | python3 -m json.tool
echo ""

echo "=== Orchestration Request ==="
curl -sk -X POST "${GATEWAY_URL}/v1/orchestrate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the sum of the first 100 prime numbers?",
    "max_turns": 10
  }' | python3 -m json.tool
echo ""
