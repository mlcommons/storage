# Submission Checker

## Overview
The submission checker is a tool designed to validate submissions against specified criteria in [Rules.md]().
**TODO:** Add this link 


## Running the Submission Checker

To run the submission checker, use the following command:

```
python -m storage.mlpstorage_py.submission_checker.main --input <submissions_folder> \
            [--version v2.0] \
            [--submitters Micron]
```

### Parameters
- `--input`: Path to the submission directory
- `--version`: MLPerf storage version
- `--submitters`: Only check the submitters provided. Comma separated.


