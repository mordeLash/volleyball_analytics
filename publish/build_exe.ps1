# build_exe.ps1
# PowerShell script to build the VolleyballAnalytics executable using PyInstaller
uv run pyinstaller --noconfirm --onedir `
  --name "VolleyballAnalytics" `
  --add-data "models;models" `
  --add-data "src;src" `
  --collect-all ultralytics `
  --collect-all openvino `
  --collect-all cv2 `
  gui.py