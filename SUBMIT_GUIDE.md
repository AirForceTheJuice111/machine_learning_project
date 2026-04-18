# Submit Guide

## Purpose

This document summarizes the practical workflow for:

- uploading your code to the remote server workspace
- manually testing the agent in the remote environment
- running `submit-test`
- running formal `submit`

It is written for the current project layout in `/workspace`.

## Key Idea

There are two different actions:

1. Upload code to `/workspace`
2. Trigger the platform submission API

Uploading code to `/workspace` is **not** the same as submitting.

## What Should Exist In `/workspace`

Your remote `/workspace` should contain at least:

- `run.sh`
- `requirements.txt`
- `agent_framework/`
- any other files needed by `run.sh`

For this project, the important files are:

- `/workspace/run.sh`
- `/workspace/agent_framework/...`
- `/workspace/requirements.txt`

## Remote Paths

Important paths in the submission environment:

- target specification: `/target/target_spec.json`
- final output: `/workspace/output.json`
- runtime log: `/workspace/results.log`

## Step 1: Connect To The Server

Example:

```bash
ssh -p 35531 root@10.176.37.31
```

After login:

```bash
cd /workspace
pwd
ls -la
```

## Step 2: Upload Or Refresh Your Code

If you need to clean the old workspace first:

```bash
cd /workspace
rm -rf agent_framework generated_cuda
rm -f run.sh requirements.txt target_spec.json output.json output.details.json output_id.txt results.log
rm -f report.*
ls -la
```

Then upload again from your local machine:

```bash
scp -P 35531 -r run.sh requirements.txt agent_framework root@10.176.37.31:/workspace/
```

If you also want to upload a local testing spec:

```bash
scp -P 35531 -r run.sh requirements.txt target_spec.json agent_framework root@10.176.37.31:/workspace/
```

## Step 3: Manual Testing In The Remote Workspace

Before using `submit-test` or `submit`, it is strongly recommended to manually verify that the code can run.

### 3.1 Check The Required Files

```bash
cd /workspace
ls -la
chmod +x run.sh
```

### 3.2 If `/target/target_spec.json` Exists

Check:

```bash
ls /target
cat /target/target_spec.json
```

If it exists, you can simulate the submission environment directly:

```bash
cd /workspace
bash run.sh
```

### 3.3 If Your Testing Spec Is In `/workspace/target/target_spec.json`

Do not use `run.sh` directly for this case. Run the evaluator manually:

```bash
cd /workspace
python3 -m agent_framework.evaluate \
  --target-spec /workspace/target/target_spec.json \
  --output /workspace/output.json \
  --skip-details-output
```

### 3.4 Check The Result

```bash
cat /workspace/output.json
tail -n 200 /workspace/results.log
```

If `results.log` is absent in manual testing, focus on `output.json`.

## Step 4: What `submit-test` Means

`submit-test` does **not** mean manually running `bash run.sh`.

It means:

- you call the platform HTTP API
- the platform launches the evaluation task
- the platform runs your `/workspace/run.sh`

So the workflow is:

1. Upload code to `/workspace`
2. Optionally test manually
3. Call `submit-test`

## Step 5: Run `submit-test`

Example:

```bash
curl -X POST http://10.176.37.31:8080/submit-test \
  -H "Content-Type: application/json" \
  -d '{ "id": "YOUR_STUDENT_ID", "gpu": 1 }'
```

Replace:

- `10.176.37.31` with the actual submission server if your course guide specifies a different one
- `YOUR_STUDENT_ID` with your own student ID

Example response:

```json
{
  "ok": true,
  "user_id": "23210240000",
  "status": "running",
  "require_gpu": true,
  "gpu_id": 0,
  "output_file": "xxxxxxxx",
  "submit_count": 0,
  "submit_limit": 2,
  "remaining_submit_count": 2
}
```

Important:

- save `output_file`

## Step 6: Query Submission Status

Use the `output_file` returned by `submit-test` or `submit`:

```bash
curl http://10.176.37.31:8080/submit_status/<output_file>
```

Example:

```bash
curl http://10.176.37.31:8080/submit_status/7f3d6d3b0d4f0b2f7a6d6d43b4b9fabc
```

Possible status values:

- `running`
- `succeeded`
- `failed`
- `killed`

## Step 7: View Output Files

Open in browser:

```text
http://10.176.37.31:8080/outputs
```

Then find your `output_file`.

You can also inspect these files in the remote workspace when available:

- `/workspace/output.json`
- `/workspace/results.log`

## Step 8: Run Formal `submit`

After `submit-test` looks good, use formal submit:

```bash
curl -X POST http://10.176.37.31:8080/submit \
  -H "Content-Type: application/json" \
  -d '{ "id": "YOUR_STUDENT_ID", "gpu": 1 }'
```

Then:

1. save `output_file`
2. query status with `/submit_status/<output_file>`
3. download or inspect the output

## Recommended Full Workflow

### Phase 1: Upload

```bash
scp -P 35531 -r run.sh requirements.txt agent_framework root@10.176.37.31:/workspace/
```

### Phase 2: Manual Check

```bash
ssh -p 35531 root@10.176.37.31
cd /workspace
chmod +x run.sh
python3 -m agent_framework.evaluate \
  --target-spec /workspace/target/target_spec.json \
  --output /workspace/output.json \
  --skip-details-output
cat /workspace/output.json
```

### Phase 3: Test Submit

```bash
curl -X POST http://10.176.37.31:8080/submit-test \
  -H "Content-Type: application/json" \
  -d '{ "id": "YOUR_STUDENT_ID", "gpu": 1 }'
```

### Phase 4: Status Query

```bash
curl http://10.176.37.31:8080/submit_status/<output_file>
```

### Phase 5: Formal Submit

```bash
curl -X POST http://10.176.37.31:8080/submit \
  -H "Content-Type: application/json" \
  -d '{ "id": "YOUR_STUDENT_ID", "gpu": 1 }'
```

## Common Confusions

### Uploading Code Is Not Submitting

These are different:

- putting files into `/workspace`
- calling `/submit-test` or `/submit`

### Manual Run Is Not Submit

These are different:

- `bash run.sh`
- `curl ... /submit-test`

Manual run is only for development verification.

### `/workspace` Is Remote

`/workspace` is the remote Linux workspace on the server or container.

It is **not** your local Windows folder.

## Final Reminder

Before formal submit, make sure:

- `/workspace/run.sh` exists
- `/workspace/output.json` can be produced
- the project reads `/target/target_spec.json`
- no local junk files or credentials are included
- after submission, prepare:
  - `/workspace/report.*`
  - `/workspace/output_id.txt`
