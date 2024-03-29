#!/usr/bin/sh

# install signatory - requires specific versions and torch to be installed first
pip install torch==1.9.0
pip install signatory==1.2.6.1.9.0

# install remaining packages
pip install argon2-cffi==21.1.0
pip install async-generator==1.10
pip install attrs==21.2.0
pip install backcall==0.2.0
pip install bleach==4.1.0
pip install cffi==1.14.6
pip install cycler==0.10.0
pip install debugpy==1.4.3
pip install decorator==5.1.0
pip install defusedxml==0.7.1
pip install entrypoints==0.3
pip install iniconfig==1.1.1
pip install ipykernel==6.4.1
pip install ipython==7.27.0
pip install ipython-genutils==0.2.0
pip install ipywidgets==7.6.5
pip install isort==5.9.3
pip install jedi==0.18.0
pip install Jinja2==3.0.1
pip install jsonschema==3.2.0
pip install jupyter==1.0.0
pip install jupyter-client==7.0.3
pip install jupyter-console==6.4.0
pip install jupyter-core==4.8.1
pip install jupyterlab-pygments==0.1.2
pip install jupyterlab-widgets==1.0.2
pip install kiwisolver==1.3.2
pip install MarkupSafe==2.0.1
pip install matplotlib==3.4.3
pip install matplotlib-inline==0.1.3
pip install mistune==0.8.4
pip install nbclient==0.5.4
pip install nbconvert==6.1.0
pip install nbformat==5.1.3
pip install nest-asyncio==1.5.1
pip install notebook==6.4.4
pip install numpy==1.21.2
pip install packaging==21.0
pip install pandas==1.3.3
pip install pandocfilters==1.5.0
pip install parso==0.8.2
pip install pexpect==4.8.0
pip install pickleshare==0.7.5
pip install Pillow==8.3.2
pip install pluggy==1.0.0
pip install prometheus-client==0.11.0
pip install prompt-toolkit==3.0.20
pip install ptyprocess==0.7.0
pip install py==1.10.0
pip install pycparser==2.20
pip install Pygments==2.10.0
pip install pyparsing==2.4.7
pip install pyrsistent==0.18.0
pip install pytest==6.2.5
pip install python-dateutil==2.8.2
pip install pytz==2021.1
pip install pyzmq==22.3.0
pip install qtconsole==5.1.1
pip install QtPy==1.11.2
pip install scipy==1.7.1
pip install seaborn==0.11.2
pip install Send2Trash==1.8.0
pip install six==1.16.0
pip install terminado==0.12.1
pip install testpath==0.5.0
pip install toml==0.10.2
pip install tornado==6.1
pip install traitlets==5.1.0
pip install typing-extensions==3.10.0.2
pip install wcwidth==0.2.5
pip install webencodings==0.5.1
pip install widgetsnbextension==3.5.1

# install jupyter kernel
python -m ipykernel install --user --name=regimedetection
