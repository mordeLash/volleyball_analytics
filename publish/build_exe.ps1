# build_exe.ps1
uv run pyinstaller --noconfirm --onedir --windowed `
  --name "VolleyballAnalytics" `
  --add-data "models;models" `
  --add-data "src;src" `
  --collect-all ultralytics `
  --collect-all cv2 `
  gui.py