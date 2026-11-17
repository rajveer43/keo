# Security policy

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/keo/) receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: <https://github.com/rajveer43/keo/security/advisories/new>. Include what you found, how to reproduce it, and which version you used.

You can expect an acknowledgement within a week. KEO is maintained by one person in their spare time, so fixes may take longer, but reports are taken seriously.

## What counts

KEO downloads a dataset, reads files, and can start worker processes for the authors' official code. Reports about unsafe file handling, unverified downloads, or code execution through a config file are in scope. A wrong benchmark result is not a security issue; please file a normal bug report for that.
