# Optional -- only created if var.alert_email is set.
# terraform apply -var="alert_email=you@example.com"
# Then click the confirmation link AWS emails you before alerts actually deliver.

resource "aws_sns_topic" "alerts" {
  count = var.alert_email != "" ? 1 : 0
  name  = "${var.project_prefix}-serving-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  count               = var.alert_email != "" ? 1 : 0
  alarm_name          = "${var.project_prefix}-serving-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  dimensions = {
    FunctionName = aws_lambda_function.serving.function_name
  }
  alarm_actions = [aws_sns_topic.alerts[0].arn]
}

resource "aws_cloudwatch_metric_alarm" "dynamodb_throttles" {
  for_each = var.alert_email != "" ? {
    online  = aws_dynamodb_table.online_features.name
    offline = aws_dynamodb_table.offline_features.name
    log     = aws_dynamodb_table.inference_log.name
  } : {}

  alarm_name          = "${var.project_prefix}-${each.key}-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ThrottledRequests"
  namespace           = "AWS/DynamoDB"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  dimensions = {
    TableName = each.value
  }
  alarm_actions = [aws_sns_topic.alerts[0].arn]
}
