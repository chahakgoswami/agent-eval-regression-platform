"""Minimal setup.py for editable installs during development."""
from setuptools import setup, find_packages

setup(
    name="agent-eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "agent-eval=agent_eval.cli:main",
        ],
    },
    python_requires=">=3.10",
)
