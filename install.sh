uv sync
uv venv .dynamic_tools_venv
source .dynamic_tools_venv/bin/activate
uv pip install -U crawl4ai
crawl4ai-setup
playwright install-deps
apt install -y npm
npm install -g @openai/codex