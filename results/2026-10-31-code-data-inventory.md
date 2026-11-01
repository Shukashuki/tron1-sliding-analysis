# TRON1 sliding-angle / XY failure：程式與資料盤點 v1

日期：2026-10-31。任務：t_980998173fdb（程式／資料盤點）。
範圍：唯讀盤點。未訓練、未模擬、未調參、未修改既有程式、設定、結果或 git 設定；唯一的執行是把既有離線重算腳本輸出到 `/tmp`，用來驗證已發表指標是否可重現（§6）。
來源版本：HEAD `a760183`，工作樹有未提交修改（7 個 M、148 個未追蹤項，見 §7-U1）。檔案 SHA-256 見同名 `.evidence.json`。

本文件補完 `results/2026-10-31-research-definitions.md` §8-1 所列「前置全專案盤點尚未完成」的缺口；指標的正式定義仍以該文件為準，本文件只記錄**現況與依據**。

---

## 1. 專案規模總覽

| 項目 | 數量／大小 | 依據 |
| --- | --- | --- |
| `scripts/*.py` | 793 檔（analyze 116、render 88、run 66、eval 48、package 42、train 40、audit 30、verify 29…） | `ls scripts/*.py` 依字首計數 |
| `tests/test_*.py` | 485 檔 | `ls tests/test_*.py` |
| `config/*.json` | 118 檔（ppo 31、contact 16、physx 15、hockey 14、nmpc 8、mppi 5、car 5…） | `ls config/*.json` |
| `results/` | 43 份 `.md` 報告、213 個子目錄，1.9 GB | `ls results`、`du -sh` |
| `outputs/` | 16 GB，被 `.gitignore` 排除 | `.gitignore` 第 4 行 `outputs/` |
| `upstream/` | 11 個 git submodule，1.5 GB | `.gitmodules`、`git submodule status` |
| `native/` | 4 個 C++ 檔（NMPC／PPO 批次 MuJoCo 預測器） | `ls native` |
| `data/` | 只有 SMPL／SMPL-X 兩個授權 zip（1.2 GB），不屬 TRON1 轉向主線 | `ls -la data`、`.gitignore` 第 7–9 行 |
| git 歷史 | 240 commits；10/14–10/22 主要是轉向／摩擦網格，10/23–10/25 是接觸／輪胎模型，10/26–10/31 是 hockey 煞停 | `git log --format='%ad %s'` |

**結論 1**：程式碼以「單一實驗一支腳本＋一支測試」的模式累積，很多是逐版本複製（例如 `ground_contact_reward_v2`…`v2t`、`analyze_cem_ground_contact_v2`…`v2o`）。和研究問題直接相關的核心模組只占少數（§2）。

---

## 2. 與研究問題直接相關的模組

研究問題（README 開頭第 3–8 行）：固定接觸摩擦下，高「行進方向對輪平面」夾角的 TRON1 輪足轉向何時出現大的水平 XY 位置誤差，並分開比較 Isaac/PhysX 與 MuJoCo。

### 2.1 參考軌跡

| 模組 | 角色 | 依據 |
| --- | --- | --- |
| `scripts/controller_references.py` | 讀取 hash 鎖定的參考目錄與分層 | docstring「Hash-locked car references for separate controller-comparison strata」 |
| `scripts/prepare_controller_reference.py` | 以明確提供的授權資產離線重建參考；hash 不符時 exit 2 | docstring；README 第 591–620 行 |
| `config/controller_references.json` | 主參考 `drift_drl_traj4`（大角度轉向）與補充 `doa_sa29_scale2`（S 形避障），`pooled_success_statistics=false` | 檔案內容 |
| `config/car_reference_lock.json` | 主參考的 CSV／motion／reference SHA-256；左輪平均絕對夾角 11.27°、峰值 27.22° | 檔案內容 |
| `scripts/hdm05_gmr.py`、`attach_human_wheel_offset.py` | 人體 GMR 重定向與腳部航向 sidecar（人體工作暫停中） | README 第 331–351 行 |

### 2.2 控制器（訓練與執行）

| 路徑 | 模組 | 依據 |
| --- | --- | --- |
| PPO | `scripts/train_beyondmimic_style.py`（273 行） | docstring；README 第 213–214 行 |
| SAC | `scripts/train_sac_tracking_isaac.py`（218 行） | docstring；README 第 215–219 行 |
| NMPC/CEM（MuJoCo） | `scripts/run_nmpc_tracking.py`（551 行）＋`nmpc_tracking_objective.py`、`nmpc_shooting.py` | docstring；README 第 220–226 行 |
| NMPC 於 Isaac plant | `scripts/run_nmpc_tracking_isaac.py`（640 行），預測器仍用 MuJoCo | docstring；README 第 223–226 行 |
| 原生預測器 | `native/nmpc_batch*.cpp`、`scripts/nmpc_native_batch*.py`，鎖定 MuJoCo 3.8.1 | README 第 242–253 行 |
| 其他試驗控制器 | MPPI（`mppi_shooting.py`）、TVLQR、SQP、adaptive MPC | README 第 37–62 行 |

### 2.3 驗證執行

| 模組 | 角色 | 依據 |
| --- | --- | --- |
| `scripts/run_car_validation.py` | 凍結 policy 的兩引擎固定 μ 網格＋俯視 XY 圖 | docstring；README 第 702–721 行 |
| `scripts/run_nmpc_validation.py` | NMPC 網格的 plan / freeze / execute 三階段；**只跑過單元測試，未產生 freeze** | docstring；README 第 277–281 行 |
| `scripts/nmpc_validation_contract.py` | 未來凍結 NMPC 網格的證據關卡 | docstring |
| `scripts/materialize_wide_friction_source.py`、`materialize_controller_source.py`、`materialize_friction_evaluator.py` | 建立隔離的評估原始碼快照（μ 至 2.0、全時程） | docstring；README 第 548–555、628–637 行 |

### 2.4 指標與分析（既有指標的實作位置）

| 模組 | 計算內容 | 依據 |
| --- | --- | --- |
| `scripts/analyze_saved_xy.py` | `path_error`：核心 [1.44,4.00] s 的參考 XY RMSE／peak、`core_complete`；`pair_error`：Isaac↔MuJoCo 路徑差；拒絕外插 | 第 56–100 行 |
| `scripts/analyze_friction_grid.py` | 由壓縮 trace 重算網格 summary，可用 `--expected-summary` 做逐值比對 | 第 56–65 行 |
| `scripts/controller_metrics.py` | `heading_signals`：β（有向）、γ（軸向）、速度 ≥0.25 m/s mask；`summarize_turn`：RMSE／peak／速度保持／低於 80% 參考速度比例／接觸 gate（count>0 且 Fz>5 N） | 第 45–200 行 |
| `upstream/tron1-learning-from-human/training/tron1_contact_slip.py` | 只看地面接觸點、以法向力加權的橫向滑移速度 | 檔案 docstring |
| `scripts/friction_cone_diagnostics.py` | 由記錄的接觸力算摩擦錐診斷；明載「不是 reward 或驗收規則」 | docstring |
| `scripts/analyze_xy_error_components.py` | 同時刻 XY 誤差拆成沿軌跡／法向分量 | docstring；README 第 159–161 行 |
| `scripts/analyze_controller_pair.py`、`analyze_policy_friction_dynamics.py` | 單一 μ 雙引擎配對稽核；重放 PPO/SAC 網格動態（不做新推論） | docstring |
| `scripts/hockey_stop_metrics.py` | 煞停距離／時間（hockey 支線，不是轉向 XY 指標） | 第 6–40 行 |

---

## 3. 設定（config）

| 設定 | 內容 | 依據 |
| --- | --- | --- |
| `config/evaluation_friction.json` | 17 點等值 μ 網格 0.01–2.00；初始 6 點；核心 [1.44,4.0] s；XY 篩選 RMSE>0.15 m 或 peak>0.30 m；訓練隨機化 static U[0.3,1.6]、dynamic U[0.3,1.2]、64 buckets；待做 (1.4,1.0)、(1.6,1.2) | 檔案內容 |
| `config/car_drift_beyondmimic_task.json` | 封存的 BeyondMimic 任務設定（含高度追蹤） | README 第 367–368 行 |
| `config/car_drift_height_smooth_task.json` | 目前比較用：移除絕對高度，改垂直加速度平滑懲罰 | README 第 122–126、644–665 行 |
| `config/car_drift_height_smooth_no_sideslip_task.json` | 無 sideslip reward 的消融版本（未追蹤） | `git status` |
| `config/nmpc_friction_development.json` | 明載 `qualified_nominal_pair_available=false`、`formal_controller_freeze_claimed=false` | research-definitions §1 已核對 |
| `config/nominal_actuator_contract.json` | 名義致動器限制 | 檔名；`tests/test_nominal_actuator_contract.py` |
| 其餘約 100 個 | 各輪實驗（ppo_ground_contact_v2…v2q、physx_tire、hockey_* 等）的單次設定 | 檔名 |

**結論 2**：研究主線的閾值、網格與核心區間集中在 `evaluation_friction.json`，且 `controller_metrics.py` 第 173、183 行把 0.15/0.30 m 寫死在程式中；兩處目前一致，但修改時需同步（§7-U6）。

---

## 4. 輸入資料與資產可用性

| 輸入 | 本機狀態 | 依據 |
| --- | --- | --- |
| 主參考 traj4 `motion.npz`（`da9945c2…`） | **有**，4 份 hash 相符，全在相鄰 repo `/home/roy/tron1-learning-from-human/`（outputs/drift-drl-turn-reference-20261013[-v2]、results/2026-10-14-car-drift/…） | 全機 `sha256sum` 比對（evidence.json） |
| 主參考 `robot_reference.npz`（`afc22f19…`） | **有**，同上 4 份 | 同上 |
| SA29 `motion.npz`（`7e6b2606…`）／`robot_reference.npz`（`79bc00a9…`） | **有**，各 6 份（本 repo `outputs/sa29_remote_dispatch_20261016_v2/…`、`outputs/reference_reproduction_20261015/…` 及相鄰 repo） | 同上 |
| 來源 CSV | 在 submodule：`upstream/drift_drl/code/ref_trajectory/traj_4.csv`、`upstream/DOA/expert_traj/…` | `config/controller_references.json` |
| 主參考 PPO checkpoint（`6ec411ef…`） | **有**：`outputs/controller_primary_ppo_mu070_20261015_v1/training/model_final.pt` | 與 `results/controller_primary_ppo_height_smooth_seed42_v1` 內 `checkpoint_sha256` 相符 |
| 主參考 SAC checkpoint（`2722b411…`） | **有**：`outputs/controller_primary_sac_mu070_20261015_v1/training/model_final.pt` | 與 `results/controller_primary_sac_height_smooth_seed42_v1` 相符 |
| SA29 PPO（`bc9123c2…`）／SAC（`1620573e…`） | **有**：`outputs/sa29_remote_dispatch_20261016_v2/training/{ppo,sac}/model_final.pt` | 與 `results/policy_sa29_friction_dynamics_20261016_v1` 相符 |
| BeyondMimic DR 基線 checkpoint（`f803a850…`，car_bm_dr_seed42） | **本機找不到**；README 說保存在遠端機器 | 全機 `*.pt` hash 搜尋無結果；README 第 447–449 行 |
| 授權 TRON1 資產（USD／MuJoCo XML） | 存在於 `/home/roy/tron1-learning-from-human/assets/robots/WF_TRON1A/`、`/home/roy/tron1-sliding-analysis-for-stopping/assets/…`；**MuJoCo XML 由 root 擁有、權限 600，目前使用者無法讀取** | `ls -l`、`sha256sum` 回報 Permission denied |
| 原生 NMPC 函式庫 | 有 `outputs/native_build_400hz_20261022_v1/libtron1_nmpc_batch_400hz.so`、`outputs/native_ground_contact_v2_20261023_v1/…`；未找到 50 Hz 版 `libtron1_nmpc_batch.so` | `find` |
| 封存的原始物理 trace（歷史基線） | **有**：`/home/roy/tron1-learning-from-human/results/2026-10-14-car-drift/`，可被 `analyze_saved_xy.py --source-root` 讀取 | §6 重算成功 |
| 相鄰 repo 版本 | tron1-learning-from-human HEAD `e0ad447`（2026-10-14），工作樹 33 項修改；tron1-sliding-analysis-for-stopping HEAD `49de035`（2026-10-11） | `git -c safe.directory=… log/status`（未改 git 設定） |

**結論 3**：轉向主線所需的 hash 鎖定參考與主參考／SA29 的 PPO、SAC checkpoint 都在本機；BeyondMimic DR 基線 checkpoint 與可讀的 MuJoCo 模型不可用，因此**物理重跑**目前不能在本機完成，只能做離線 trace 重算。

---

## 5. 輸出與既有結果（依研究線分類）

| 研究線 | 代表結果 | 主要數值（已核對） | 依據 |
| --- | --- | --- | --- |
| 歷史固定 μ=0.15 policy 基線 | `results/car_baseline/` | μ=0.70：Isaac 0.0985 m／MuJoCo 0.1030 m；μ=1.00：0.1502／0.1510 m；μ=0.05 截尾 | `summary.json`（§6 重算相符） |
| BeyondMimic DR，6 點／17 點／全時程 | `results/car_bm_dr_seed42/`、`car_bm_dr_wide_seed42/`、`car_bm_dr_full_horizon_seed42/` | 17 點：Isaac 12 個低於篩選＋5 個截尾；MuJoCo 13＋4 | `summary.json` 逐列統計 |
| Height-smooth PPO 17 點 | `results/controller_primary_ppo_height_smooth_seed42_v1/` | 低於篩選：Isaac 13/17、MuJoCo 11/17 | `summary.json` |
| Height-smooth SAC 17 點 | `results/controller_primary_sac_height_smooth_seed42_v1/` | 低於篩選：0/17（兩引擎） | `summary.json` |
| NMPC 開發（未凍結） | `results/nmpc_primary_friction_pair_development_20261016_v1/` 等 | README：每引擎 10/17，同 μ 兩引擎皆過 8/17 | README 第 82–84 行 |
| SA29 補充分層 | `results/policy_sa29_friction_dynamics_20261016_v1/`、`sa29_controller_comparison_20261016_v1/` | README：PPO 12/17 Isaac、11/17 MuJoCo | README 第 100–101 行 |
| CEM sideslip × 已知 μ | `results/nmpc_beta_friction_factorial[_17]_20261017_v1/` | 136 個分析 cell（132 個不重複 episode） | 報告 md |
| 接觸／輪胎模型研究（10/23–10/25） | `ground_contact_v2*`、`physx_tire_*`、`contact_*` 等約 90 個目錄 | — | `ls results` |
| Hockey 煞停（10/26–10/31） | `hockey_*` 約 20 個目錄，最新 `hockey_contact_model_framework_20261031_v1` | 4 個名義 rollout，XY RMSE 2.3–2.8 cm | 該目錄 README |

**結論 4**：與「轉向 XY 失敗 × 摩擦」直接相關的完整 17 點網格證據有 4 組：BM-DR、height-smooth PPO、height-smooth SAC、NMPC 開發版；只有 PPO/SAC 有對應的本機 checkpoint。所有結果都是單一 seed、確定性單 episode。

**結論 5**：近 6 天的提交（`git log` 10/26 起）全部屬於 hockey 煞停支線，README 沒有提到 hockey（`grep -i hockey README.md` 無結果）；研究定義草案 §1 也明確把 hockey 排除在範圍外。主線與支線的關係需要確認（§7-U2）。

---

## 6. 實際驗證（離線重算，輸出到 `/tmp`）

環境：系統 `python3` 3.12.3 沒有 numpy／pytest；使用既有 conda 環境 `retarget_sim`（numpy 2.4.6、scipy 1.17.1、matplotlib 3.10.9、mujoco 3.9.0，**沒有 pytest**）。沒有安裝任何套件。

| 命令（工作目錄 = repo 根目錄） | 結果 |
| --- | --- |
| `analyze_saved_xy.py --trace results/car_baseline/xy_trace.npz --output-dir /tmp/inv_replay.*` | exit 0；重算的 `summary.json` 與已發表版本**完全相等**（dict 比對 True） |
| `analyze_saved_xy.py --source-root /home/roy/tron1-learning-from-human --output-dir /tmp/inv_srcreplay` | exit 0；從**封存原始物理 trace** 重建的 `summary.json`、`xy_trace.provenance.json` 與已發表版本 byte 相同（`cmp`）；`xy_trace.npz` SHA-256 相同（`30a10203…`） |
| `analyze_friction_grid.py --trace … --expected-summary …`（`car_bm_dr_seed42`、`car_bm_dr_wide_seed42`、`car_bm_dr_full_horizon_seed42`） | 三者 exit 0（6／17／7 個 case）；腳本第 65 行在不相等時會報錯，因此代表與已發表 summary 完全相等 |
| `pytest` | **未執行**：所有可用環境都沒有 pytest（`.venv`、`mujoco_try`、`retarget_sim` 均缺） |

**結論 6**：歷史基線的 XY 指標可從原始 trace 端到端 byte 級重現；BM-DR 三組網格可從壓縮 trace 重現。PPO/SAC/NMPC 開發結果本次未重算。

---

## 7. 未知事項與風險

| # | 未知／風險 | 影響 | 建議的確認方式 |
| --- | --- | --- | --- |
| U1 | 工作樹有未提交修改：`README.md`（+23 行）、`run_nmpc_tracking.py`（+20）、`run_nmpc_tracking_isaac.py`（+12）等 7 檔，及 148 個未追蹤項（含 11 個 config、多份 10/16–10/21 報告） | HEAD 不能代表產生結果的程式碼版本 | 由使用者決定是否提交；基線確認時以檔案 hash 為準 |
| U2 | Hockey 煞停支線與轉向 XY 主線的關係不明（README 未提及，研究定義排除） | 基線與研究問題的範圍 | 由使用者確認 hockey 是否納入本研究 |
| U3 | BeyondMimic DR 基線 checkpoint（`f803a850…`）不在本機 | 無法在本機重跑該 policy | 確認遠端存放位置與存取權限 |
| U4 | MuJoCo `robot.xml` 由 root 擁有、權限 600，目前使用者無法讀取；`export_model_sha256 30c8bb94…` 是否對應其中某檔未驗證 | 物理重跑與參考重建 | 由使用者調整權限或提供可讀副本後比對 hash |
| U5 | 沒有任何環境安裝 pytest，485 個測試本次未執行；也未確認 Isaac Lab 2.3 環境是否存在 | 程式正確性只有閱讀證據＋§6 的重算 | 經授權後在專用環境安裝 pytest 並執行 `tests/test_xy_analysis.py`、`tests/test_controller_metrics.py` |
| U6 | XY 閾值同時存在於 `evaluation_friction.json` 與 `controller_metrics.py` 第 173、183 行（寫死） | 修改閾值時可能不一致 | 基線確認時固定單一來源 |
| U7 | `analyze_saved_xy.path_error` 只檢查核心結束時間、未檢查起點與丟幀（research-definitions §3 已指出） | 截尾判定可能偏寬 | 加入 preflight 前不改寫既有結果 |
| U8 | 正式 NMPC freeze 未產生；traj4 重建結果與 byte lock 不符（差 ≤3.56e-13），原因未知 | NMPC 不能當凍結基線；主參考只能用既有凍結檔 | README 第 613–620 行；維持現況 |
| U9 | 相鄰 repo `tron1-learning-from-human` 工作樹有 33 項修改，與 `upstream/` 中 pinned submodule（`03450544…`）不同 | 原始 trace 的程式版本 lineage | 以 `xy_trace.provenance.json` 的 hash 為準 |
| U10 | 不同 `outputs/` 大量為快照 tar.gz（16 GB），未逐一核對內容與 manifest | 部分歷史證據的完整性 | 依需要針對基線候選逐一驗證 |
| U11 | Distinct static/dynamic μ（1.4/1.0、1.6/1.2）在轉向主線是否已支援未確認（hockey 支線 10/28 有拆分 static/dynamic 的提交） | 網格擴充 | 基線確認階段核對 evaluator 讀回 |

---

## 8. 驗收對照

| 任務要求 | 段落 | 狀態 |
| --- | --- | --- |
| 相關模組 | §2 | 完成，每項附檔案／行號 |
| 輸入資料 | §4 | 完成，含全機 hash 比對 |
| 設定 | §3 | 完成 |
| 輸出 | §1、§5 | 完成（outputs 快照內容見 U10） |
| 既有指標 | §2.4、§5、§6 | 完成，其中 4 組已重算驗證 |
| 每項結論附依據 | 各表「依據」欄、evidence.json | 完成 |
| 未知事項 | §7 | 11 項 |
