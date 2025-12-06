#!/usr/bin/env pwsh
Set-Location -Path (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent)
python -u run_train.py
