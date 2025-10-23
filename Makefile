# ЗАМЕНИТЕ весь Makefile на этот (или добавьте недостающие таргеты)

ZIP := tick-coach-agent.zip
FUNC := tick-coach-agent
ENTRY := index.main
RUNTIME := python312
MEM := 128m
TIMEOUT := 300s
SA := ajehjub0fqvlq9u84lup
OPENROUTER_BASE_URL := https://openrouter.ai/api/v1
GROQ_BASE_URL := https://api.groq.com
AGENTROUTER_BASE_URL := https://agentrouter.org/v1
LANGSMITH_TRACING := true
LANGSMITH_PROJECT := "tick_coach_agent"

export OPENROUTER_BASE_URL
export GROQ_BASE_URL
export AGENTROUTER_BASE_URL
export OPENROUTER_API_KEY
export GROQ_API_KEY
export AGENTROUTER_API_KEY
export LANGSMITH_API_KEY
export PROXY_HOST
export PROXY_PORT
export PROXY_USER
export PROXY_PASS


.PHONY: build-zip clean deploy

build-zip:
	rm -f $(ZIP)
	# requirements.txt без dev-группы, комментариев, заголовка и локальных пакетов
	uv export --format requirements-txt --frozen --no-dev --no-annotate --no-header --no-hashes --no-emit-local -o requirements.txt
	# Кладём ТОЛЬКО runtime: папку agents, index.py и requirements.txt
	zip -9 -r $(ZIP) agents index.py requirements.txt


# === PROXY compose ===
PROXY_SCHEME := http
PROXY_AUTH   := $(if $(PROXY_USER),$(PROXY_USER):$(PROXY_PASS)@,)
PROXY_URL    := $(if $(PROXY_HOST),$(PROXY_SCHEME)://$(PROXY_AUTH)$(PROXY_HOST):$(PROXY_PORT),)
export PROXY_URL


ENV_ARGS = "OPENROUTER_API_KEY=$$OPENROUTER_API_KEY,GROQ_API_KEY=$$GROQ_API_KEY,AGENTROUTER_API_KEY=$$AGENTROUTER_API_KEY,LANGSMITH_API_KEY=$$LANGSMITH_API_KEY,OPENROUTER_BASE_URL=$$OPENROUTER_BASE_URL,GROQ_BASE_URL=$$GROQ_BASE_URL,AGENTROUTER_BASE_URL=$$AGENTROUTER_BASE_URL,PROXY_HOST=$$PROXY_HOST,PROXY_PORT=$$PROXY_PORT,PROXY_USER=$$PROXY_USER,PROXY_PASS=$$PROXY_PASS,PROXY_URL=$$PROXY_URL"

deploy-yc: build-zip
	yc serverless function version create \
	  --function-name $(FUNC) \
	  --runtime $(RUNTIME) \
	  --entrypoint $(ENTRY) \
	  --memory $(MEM) \
	  --execution-timeout $(TIMEOUT) \
	  --service-account-id $(SA) \
	  --source-path ./$(ZIP) \
	  --format json > func_ver.json \
	  --environment $(ENV_ARGS)

clean:
	rm -f $(ZIP) func_ver.json

deploy: deploy-yc
	clean

git:
	@if [ -z "$(M)" ]; then echo 'ERROR: set M, e.g. make git M="feat: deploy function"'; exit 1; fi
	git add -A
	git commit -m "$(M)"
	git push origin main