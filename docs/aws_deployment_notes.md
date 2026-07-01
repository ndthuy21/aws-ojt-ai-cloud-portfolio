# AWS Deployment Notes

This project is intentionally small enough to deploy on an AWS Free Tier-style
learning setup. The goal is not to claim production experience, but to show a
clear and safe deployment path.

## Target Architecture

```text
Developer laptop
  -> GitHub repository
  -> EC2 Linux instance
  -> Python HTTP inference API
  -> CloudWatch logs
  -> S3 bucket for model/data artifacts
```

## Suggested AWS Services

- **EC2**: run the Python HTTP inference API on a small Linux instance.
- **S3**: store model artifacts, sample datasets, and generated reports.
- **IAM**: create a least-privilege role/policy for S3 read-only access.
- **CloudWatch**: collect app logs and basic instance metrics.
- **AWS Budgets / Billing alerts**: prevent unexpected cost.

## Safe Free Tier Practice Checklist

- Use a small instance type only for short practice sessions.
- Stop or terminate the EC2 instance after testing.
- Keep S3 objects small and delete unused artifacts.
- Set a billing alert before creating resources.
- Avoid public write access on S3 buckets.
- Do not commit AWS credentials to GitHub.

## Example EC2 Deployment Flow

```bash
sudo yum update -y
sudo yum install -y git python3
git clone https://github.com/ndthuy21/aws-ojt-ai-cloud-portfolio.git
cd aws-ojt-ai-cloud-portfolio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
python cloud_ready_ml_api/app.py --host 0.0.0.0 --port 8000
```

Security group practice:

- Allow SSH only from your IP.
- Allow port `8000` only for temporary testing.
- Close public access after the demo.

