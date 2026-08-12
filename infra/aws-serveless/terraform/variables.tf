variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_prefix" {
  description = "Prefix applied to every resource name/table/bucket in this stack."
  type        = string
  default     = "churn"
}

variable "ecr_image_tag" {
  description = "Tag of the serving image to deploy. The CI/CD workflow updates the running function's image via `aws lambda update-function-code` after each push to aws-serverless, so this only matters for the FIRST `terraform apply` (before any image has been pushed, point it at a tag you've already built once by hand)."
  type        = string
  default     = "latest"
}

variable "alert_email" {
  description = "Email to receive CloudWatch alarm notifications (Lambda errors, DynamoDB throttling). Leave blank to skip creating the SNS topic/subscription entirely."
  type        = string
  default     = ""
}
