# run_eval.ps1（フラット配置用）
# 必要なら下行をあなたの venv に変更
$py = "python"

& $py -V
& $py -m pip install -r requirements.txt

& $py .\tch_eval_classify.py `
  --faq_xlsx .\faq.xlsx `
  --input_csv .\test.csv `
  --st_model intfloat/multilingual-e5-large-instruct `
  --threshold 0.905 `
  --out_dir .\out

Write-Host "Done. See .\out"
