!macro tauri_uninstall
  DetailPrint "清理 StrategyForge 运行期数据..."
  RMDir /r "$LOCALAPPDATA\StrategyForge\data"
  DetailPrint "运行期数据已清理"
!macroend
