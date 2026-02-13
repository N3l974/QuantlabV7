# QuantLab V7 — Commandes rapides
# Usage: make <target>

VPS_HOST    ?= 51.255.201.207
VPS_USER    ?= ubuntu
SSH_KEY     ?= ~/.ssh/id_ed25519
LOG_DIR     ?= /home/ubuntu/quantlab-deploy/runtime/logs/v5c-highrisk-paper
HOURS       ?= 24

# ---------- Paper reporting ----------

paper-report: ## Fetch VPS logs and print daily paper report
	python3 scripts/paper_daily_report_remote.py \
		--vps-host $(VPS_HOST) \
		--vps-user $(VPS_USER) \
		--ssh-key $(SSH_KEY) \
		--remote-log-dir $(LOG_DIR) \
		--hours $(HOURS)

paper-report-48h: ## Same but 48h window
	$(MAKE) paper-report HOURS=48

vps-logs: ## Quick ls of log files on VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) "ls -lh $(LOG_DIR)"

vps-status: ## Docker status on VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) "cd ~/quantlab-deploy && docker compose -f docker-compose.portfolio.yml ps"

vps-tail: ## Tail last 50 lines of container logs
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) "docker logs --tail 50 v5c-highrisk-paper"

# ---------- Deploy ----------

deploy: ## Trigger GitHub Actions deploy
	gh workflow run deploy-portfolio.yml --ref main -f portfolio_id=v5c-highrisk-paper -f image_tag=latest
	@echo "Watch with: make deploy-watch"

deploy-watch: ## Watch latest deploy run
	$(eval RUN_ID := $(shell gh run list --workflow=deploy-portfolio.yml --limit 1 --json databaseId --jq '.[0].databaseId'))
	gh run watch $(RUN_ID)

# ---------- Dev ----------

test: ## Run tests
	pytest tests/ -v

ingest: ## Ingest Binance data
	python main.py ingest

# ---------- Help ----------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
.PHONY: paper-report paper-report-48h vps-logs vps-status vps-tail deploy deploy-watch test ingest help
