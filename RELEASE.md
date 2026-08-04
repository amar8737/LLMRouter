# Releasing to PyPI

Steps to publish a release to PyPI via GitHub Actions:

1. Create a PyPI API token: go to https://pypi.org/manage/account/#api-tokens and create a token for this project.
2. In your GitHub repository, add the token as a secret named `PYPI_API_TOKEN` (Repository Settings → Secrets).
3. Tag a release and push to GitHub. Example:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin --tags
```

The workflow `.github/workflows/publish.yml` will build and publish the package when a tag matching `v*` is pushed.

Local publish (alternative):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip build twine
python3 -m build
python3 -m twine upload dist/*
```
