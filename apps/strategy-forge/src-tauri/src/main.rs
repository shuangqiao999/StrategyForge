#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

static BACKEND_CHILD: Mutex<Option<Child>> = Mutex::new(None);

#[cfg(target_os = "windows")]
fn apply_no_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn apply_no_window(_cmd: &mut Command) {}

fn launch_backend() {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();

    let backend_paths = [
        exe_dir.join("strategy-forge-backend/strategy-forge-backend.exe"),
        exe_dir.join("strategy-forge-backend.exe"),
    ];

    let mut child_opt: Option<Child> = None;
    for path in &backend_paths {
        if path.exists() {
            let mut cmd = Command::new(path);
            cmd.arg("serve");
            // 固定后端工作目录与数据目录到后端 exe 同级，避免 cwd 漂移
            // 导致 forge_config.json 与 Kuzu/LanceDB 数据分散到不同位置。
            if let Some(backend_dir) = path.parent() {
                cmd.current_dir(backend_dir);
                // 运行期数据：%LOCALAPPDATA%\StrategyForge\data（卸载不丢、无 UAC 虚拟化）
                let local_data = std::env::var("LOCALAPPDATA").unwrap_or_default();
                if !local_data.is_empty() {
                    cmd.env("FORGE_DATA_DIR",
                            std::path::PathBuf::from(&local_data)
                                .join("StrategyForge").join("data"));
                }
                // 内置规则包：安装目录下的 data/rule（只读，随版本更新）
                cmd.env("FORGE_RULE_DIR", backend_dir.join("data").join("rule"));
            }
            apply_no_window(&mut cmd);
            child_opt = cmd.spawn().ok();
            break;
        }
    }
    if child_opt.is_none() {
        eprintln!("[StrategyForge] Backend exe not found. Tried:");
        for path in &backend_paths {
            eprintln!("  - {}", path.display());
        }
    }
    if let Ok(mut guard) = BACKEND_CHILD.lock() {
        *guard = child_opt;
    }
}

fn kill_backend() {
    if let Ok(mut guard) = BACKEND_CHILD.lock() {
        if let Some(ref mut child) = *guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *guard = None;
    }
}

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};

    let show = MenuItem::with_id(app, "show", "显示窗口", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    let icon = app.default_window_icon().cloned().ok_or("missing window icon")?;

    TrayIconBuilder::with_id("main_tray")
        .icon(icon)
        .tooltip("StrategyForge — 战略决策推演")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app_handle, event| match event.id.as_ref() {
            "quit" => {
                kill_backend();
                app_handle.exit(0);
            }
            "show" => {
                if let Some(w) = app_handle.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.unminimize();
                    let _ = w.set_focus();
                }
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.unminimize();
                    let _ = w.set_focus();
                }
            }
        })
        .build(app)?;

    Ok(())
}

fn main() {
    launch_backend();

    let app = tauri::Builder::default()
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .setup(|app| {
            setup_tray(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running StrategyForge");

    app.run(|_handle, event| {
        if let tauri::RunEvent::Exit = event {
            kill_backend();
        }
    });
}
