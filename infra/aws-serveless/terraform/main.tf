terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# DynamoDB: feature store (replaces the local feature_store.sqlite)
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "online_features" {
  name         = "${var.project_prefix}-online-features"
  billing_mode = "PAY_PER_REQUEST" # on-demand -- no capacity to plan/pay for at rest
  hash_key     = "entity_id"

  attribute {
    name = "entity_id"
    type = "S"
  }
}

resource "aws_dynamodb_table" "offline_features" {
  name         = "${var.project_prefix}-offline-features"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "entity_id"
  range_key    = "ts"

  attribute {
    name = "entity_id"
    type = "S"
  }
  attribute {
    name = "ts"
    type = "S"
  }
}

resource "aws_dynamodb_table" "inference_log" {
  name         = "${var.project_prefix}-inference-log"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "log_shard"
  range_key    = "sk"

  attribute {
    name = "log_shard"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }
}

# ---------------------------------------------------------------------------
# S3: model registry (replaces the local registry/ directory)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "registry" {
  bucket = "${var.project_prefix}-registry-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "registry" {
  bucket                  = aws_s3_bucket.registry.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "registry" {
  bucket = aws_s3_bucket.registry.id
  versioning_configuration {
    status = "Enabled" # cheap insurance against an accidental overwrite of a pointer file
  }
}

# ---------------------------------------------------------------------------
# ECR: serving image repository
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "serving" {
  name                 = "${var.project_prefix}-serving"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_ecr_lifecycle_policy" "serving" {
  repository = aws_ecr_repository.serving.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 3 -- ECR storage is billed per GB-month"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM: Lambda execution role, scoped to exactly the tables/bucket above
# ---------------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_prefix}-serving-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_data_access" {
  name = "${var.project_prefix}-serving-data-access"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query",
        ]
        Resource = [
          aws_dynamodb_table.online_features.arn,
          aws_dynamodb_table.offline_features.arn,
          aws_dynamodb_table.inference_log.arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.registry.arn, "${aws_s3_bucket.registry.arn}/*"]
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda: serving API (FastAPI + Mangum, packaged via Dockerfile.lambda)
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "serving" {
  name = "/aws/lambda/${var.project_prefix}-serving"
  # Short retention on purpose -- CloudWatch Logs storage with the default
  # "never expire" setting is the single easiest line item to let creep past
  # a $5/month budget without anyone noticing for months.
  retention_in_days = 14
}

resource "aws_lambda_function" "serving" {
  function_name = "${var.project_prefix}-serving"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.serving.repository_url}:${var.ecr_image_tag}"

  # xgboost + pandas import is slow enough on cold start to want the same
  # headroom lightgbm needed in Project 1 (raised there from 512/10 after
  # cold-start timeouts). Revisit downward if cold starts prove fast in
  # practice -- memory is the dominant Lambda cost lever.
  memory_size = 1024
  timeout     = 30

  environment {
    variables = {
      DYNAMODB_ONLINE_TABLE  = aws_dynamodb_table.online_features.name
      DYNAMODB_OFFLINE_TABLE = aws_dynamodb_table.offline_features.name
      DYNAMODB_LOG_TABLE     = aws_dynamodb_table.inference_log.name
      S3_REGISTRY_BUCKET     = aws_s3_bucket.registry.bucket
      AWS_REGION             = var.aws_region
      CANARY_PCT             = "0"
    }
  }

  depends_on = [aws_cloudwatch_log_group.serving]
}

resource "aws_lambda_function_url" "serving" {
  function_name      = aws_lambda_function.serving.function_name
  authorization_type = "NONE" # same public-Function-URL approach used for Project 1's two scoring Lambdas
}

output "function_url" {
  value = aws_lambda_function_url.serving.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.serving.repository_url
}

output "registry_bucket" {
  value = aws_s3_bucket.registry.bucket
}

output "lambda_role_arn" {
  value = aws_iam_role.lambda_exec.arn
}
