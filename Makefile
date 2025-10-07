# ЗАМЕНИТЕ весь Makefile на этот (или добавьте недостающие таргеты)

ZIP := tick-coach-agent.zip
FUNC := tick-coach-agent
ENTRY := index.main
RUNTIME := python311
MEM := 128m
TIMEOUT := 10s
SA := ajehjub0fqvlq9u84lup

.PHONY: build-zip clean deploy

build-zip:
	rm -f $(ZIP)
	# упакуем код функции; requirements.txt добавится если есть
	@if [ -f requirements.txt ]; then zip -9 $(ZIP) index.py requirements.txt; else zip -9 $(ZIP) index.py; fi

deploy: build-zip
	yc serverless function version create \
	  --function-name $(FUNC) \
	  --runtime $(RUNTIME) \
	  --entrypoint $(ENTRY) \
	  --memory $(MEM) \
	  --execution-timeout $(TIMEOUT) \
	  --service-account-id $(SA) \
	  --source-path ./$(ZIP) \
	  --format json > func_ver.json

clean:
	rm -f $(ZIP) func_ver.json
