---
name: GitHub Actions Local Testing
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers: [act]
---

# GitHub Actions Local Testing with Act

This microagent helps you run GitHub Actions locally using the 'act' CLI tool to test and fix issues before pushing commits, reducing the number of pushes needed to get a PR ready to merge.

## Prerequisites

- Install the 'act' CLI tool: `curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash`
- Docker must be installed and running
- Ensure your GitHub Actions workflows are in `.github/workflows/`

## Usage

When triggered by the keyword "act", this microagent will:

1. **Check for act installation**: Verify that 'act' is installed and available
2. **List available workflows**: Show all GitHub Actions workflows in the repository
3. **Run workflows locally**: Execute workflows using 'act' to simulate GitHub Actions
4. **Analyze results**: Identify any failures, errors, or issues
5. **Provide fixes**: Suggest and implement fixes for any detected issues
6. **Re-run tests**: Verify that fixes resolve the issues

## Common Workflows to Test

For this repository, the main workflow to test is:
- `.github/workflows/openhands-resolver.yml`

## Act Commands

- `act -l`: List all available jobs
- `act`: Run all jobs
- `act -j <job-name>`: Run a specific job
- `act -W <workflow-file>`: Run a specific workflow
- `act --dry-run`: Show what would be executed without running

## Error Handling

- If Docker is not running, start Docker service
- If 'act' is not installed, provide installation instructions
- If workflows fail, analyze logs and provide specific fix suggestions
- Handle missing secrets or environment variables by providing guidance

## Limitations

- Some GitHub Actions features may not work locally (e.g., GitHub API calls)
- Certain secrets and contexts may need to be mocked
- Network-dependent actions might behave differently locally

## Example Usage

```
User: "act"
Assistant: I'll run the GitHub Actions locally using act to test the workflows before you push.
```

This helps ensure your commits pass CI/CD checks on the first push attempt.
