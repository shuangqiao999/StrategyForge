!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
    DetailPrint "清理 StrategyForge 运行期数据..."
    RMDir /r "$LOCALAPPDATA\StrategyForge\data"
    DetailPrint "StrategyForge 运行期数据已清理"
  ${EndIf}
!macroend
