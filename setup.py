from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="facial-recognition-fl",
    version="1.0.0",
    author="Healthcare AI Team",
    author_email="team@healthcare-ai.com",
    description="Production-ready federated learning system for bias-free facial recognition in healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/healthcare-ai/facial-recognition-fl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.11.0",
            "flake8==6.1.0",
            "mypy==1.7.0",
            "isort==5.12.0",
            "pre-commit==3.5.0",
        ],
        "docs": [
            "sphinx==7.2.6",
            "sphinx-rtd-theme==2.0.0",
            "sphinx-autodoc-typehints==1.25.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "fl-server=server.federated_server:main",
            "fl-client=client.federated_client:main",
        ],
    },
)