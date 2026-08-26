@echo off
set "TARGET_BAT=%~dp0run_app.bat"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\PEGASUS Video Retrieval.lnk"

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%TARGET_BAT%'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'PEGASUS AI Challenge 2026'; $Shortcut.Save()"

echo [OK] Da tao Shortcut 'PEGASUS Video Retrieval' ngoai Desktop!
