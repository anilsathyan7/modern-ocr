# install jupyter lab globally
uv tool install jupyterlab --with pip

# create a new project with uv
uv init my-new-project
cd my-new-project

# start jupyter lab
uv run jupyter-lab

uv run jupyter-lab

# install paddleocr
pip install paddlepaddle-gpu==3.3.1 -i "https://www.paddlepaddle.org.cn/packages/stable/cu130/"
pip3 install --upgrade paddleocr[all]


# optionally install jupyter ai for agents
Link: https://jupyter-ai.readthedocs.io/en/latest/getting-started.html

uv pip install jupyter-ai
curl -fsSL https://chatgpt.com/codex/install.sh | sh
sudo apt install nodejs
sudo npm install -g @zed-industries/codex-acp
