# Sourced by workflow steps that run Claude Code. Uses prepaid Anthropic credit first:
# when ANTHROPIC_DIRECT_KEY is set and a one-token call with the job's model succeeds, Claude Code
# talks to Anthropic directly. Otherwise (no key, credit used up, outage) nothing changes and the
# step runs through OpenRouter as before. The choice is made once per job.
#
# Reads:   ANTHROPIC_DIRECT_KEY, OPENROUTER_MODEL (e.g. anthropic/claude-sonnet-5)
# Exports: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, CHARTS_AGENT_TOKEN,
#          ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL, CLAUDE_CODE_SUBAGENT_MODEL, OPENROUTER_MODEL
claude_upstream() {
  local key="${ANTHROPIC_DIRECT_KEY:-}"
  unset ANTHROPIC_DIRECT_KEY
  local model="${OPENROUTER_MODEL:-anthropic/claude-sonnet-5}"
  case "$model" in anthropic/claude-*) ;; *) echo "[upstream] openrouter (model $model is not an Anthropic model)"; return 0 ;; esac
  if [ -z "$key" ]; then echo "[upstream] openrouter (no Anthropic key)"; return 0; fi
  local direct="${model#anthropic/}"
  direct="${direct//./-}"
  local status
  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://api.anthropic.com/v1/messages \
    -H "x-api-key: $key" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
    -d "{\"model\":\"$direct\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}" || echo 000)
  if [ "$status" != "200" ]; then echo "[upstream] openrouter (Anthropic check returned $status)"; return 0; fi
  export ANTHROPIC_BASE_URL=https://api.anthropic.com
  export ANTHROPIC_API_KEY="$key"
  export ANTHROPIC_AUTH_TOKEN=""
  export CHARTS_AGENT_TOKEN=""
  export OPENROUTER_MODEL="$direct"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$direct" ANTHROPIC_DEFAULT_SONNET_MODEL="$direct" ANTHROPIC_DEFAULT_HAIKU_MODEL="$direct" CLAUDE_CODE_SUBAGENT_MODEL="$direct"
  echo "[upstream] anthropic ($direct)"
}
claude_upstream
