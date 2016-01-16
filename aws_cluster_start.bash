#!/bin/bash
aws emr create-cluster --name "stmpy" --release-label emr-4.2.0 \
--use-default-roles --ec2-attributes KeyName=acoppola \
--applications Name=Spark Name=Hadoop \
--instance-count 3 --instance-type m1.medium \
--bootstrap-action Path="s3://aws-logs-770047837869-us-east-1/elasticmapreduce/bootstrap-actions/stmpy_bootstrap.bash"