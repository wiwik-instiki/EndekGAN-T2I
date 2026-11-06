# Publish through VS Code, without PowerShell

1. Open the complete EndekGAN_GITHUB_COMPLETE_WORKFLOW_20261106 folder in VS Code, not the parent experimental folder.
2. Review README.md, docs/DATA_RIGHTS.md, the source notebook cells, and repository visibility with the paper coauthors. Consider a PRIVATE repository until publication/rights are cleared.
3. In VS Code's Source Control icon, choose Initialize Repository. Make sure local_runs, private data, model checkpoints, entire raw image folders, .env files, and ZIPs are NOT staged.
4. Stage approved files, enter a commit message, and Commit.
5. Use Publish to GitHub in Source Control and select the chosen PRIVATE/PUBLIC visibility.
6. Verify on github.com that the master notebook, results, figures and docs render and that no textile image data or model weights were published.

The ZIP in the parent directory is an archive for transport and backup. GitHub's web Upload Files screen does NOT automatically unzip archives; upload the extracted contents or publish using VS Code Source Control.
This package has NOT been uploaded or pushed by ChatGPT.
