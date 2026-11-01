# TRON1 sliding-angle / XY failure：基線確認 v1

日期：2026-10-31。任務：t_9662edb0f4b0（基線確認）。
前置文件：[程式／資料盤點](2026-10-31-code-data-inventory.md)、[研究定義](2026-10-31-research-definitions.md)。指標與 F/P/C/I 判定沿用研究定義 §3，本文件不重新定義。
範圍：只做離線驗證。沒有訓練、物理模擬、調參，也沒有修改既有程式、設定或結果。安裝的套件只放在 `/tmp` 下的臨時 venv；臨時 git worktree 用完已移除。檔案 SHA-256 與命令記錄在同名 [`.evidence.json`](2026-10-31-baseline-confirmation.evidence.json)。
來源版本：HEAD `a760183`，工作樹未提交（7 M、約 150 個未追蹤項）。

---

## 0. 結論摘要

1. **目前不存在可以直接拿來做實驗的「正式凍結基線」**（研究定義 §5 的第 C 層）。現有的都是開發用證據，而且都只有單一 seed、每個 μ 一個確定性 episode。
2. 主線（traj4 大角度轉向）**最適合當參考基線的候選**是 **height-smooth PPO seed42**：checkpoint、task profile、參考目錄、trainer 的 hash 都和目前檔案一致，17 點網格 summary 可以從壓縮 trace 完全重算出來。它只能當「描述性比較基線」，要升格成正式基線，需要使用者決定 §6 的 D1–D3。
3. **沒有資料切分**：訓練與評估用同一條參考軌跡；17 點 μ 網格全部都已公開看過，只能算開發集，沒有 held-out。依 README 描述推導，訓練時的**有效**接觸摩擦範圍比名目 DR 範圍窄很多（§3.2）。
4. 可重現性：XY 指標的離線重算在兩條研究線都做到 **byte 級重現**。完整測試套件在鎖定版本的環境中是 8345 通過 / 107 失敗 / 38 錯誤。失敗的原因可以分成 5 類，**沒有一類影響主線 XY 指標**（主線指標測試 73/73 通過）。但其中兩類代表**已提交的程式碼已經偏離歷史實驗鎖定的來源**（§5.3）。
5. 範圍衝突仍未解決：最近 6 天的工作全部在 hockey 煞停支線，但研究定義把 hockey 排除在範圍外（§1）。

---

## 1. 範圍與兩條研究線

| 研究線 | 狀態 | 本文件處理方式 |
| --- | --- | --- |
| 主線：`drift_drl_traj4` 大角度轉向 × 固定 μ × Isaac/MuJoCo | 研究定義 v1 的範圍 | 完整確認（§2–§5） |
| 補充分層：`doa_sa29_scale2` S 形避障 | 分開報告，禁止合併成功率 | 列出基線，沿用同一流程 |
| 支線：hockey 煞停（10/26–10/31，最新 commit `a760183`） | README 沒有提到；研究定義 §1 排除 | 列出現有基線與可重現性，**不判定是否納入**（決策項 D1） |

**範圍衝突**：專案指示是「sliding-angle / XY failure analysis」；研究定義（t_16c65cbedb15）把 hockey 排除；但 git 歷史顯示目前實際投入的是 hockey。這三者衝突，本文件不自行調解，只記錄為 D1。

---

## 2. 現有基線清單與判定

### 2.1 主線 traj4（核心 [1.44, 4.00] s，XY 篩選 RMSE>0.15 m 或 peak>0.30 m）

| 基線 | 訓練／設定 | 17 點結果（低於篩選） | 本機可用性 | 重現驗證 | 判定 |
| --- | --- | --- | --- | --- | --- |
| A. 歷史 μ=0.15 固定摩擦 policy（`results/car_baseline`） | 舊 actor，沒有 BM 摩擦 DR；只有 3 個 μ | μ=0.70／1.00 完整；0.05 截尾 | 原始 trace 在相鄰 repo | ✅ summary dict 相等（本次）；原始 trace 端到端 byte 相同（盤點 §6） | 只能用來檢查**分析相容性**，不是比較基線 |
| BM-DR seed42（`car_bm_dr_*`） | BeyondMimic DR，舊 task profile（有高度追蹤） | Isaac 12+5 截尾；MuJoCo 13+4 截尾 | **checkpoint `f803a850…` 不在本機** | ✅ 三組網格 summary 重算相等（盤點 §6） | 歷史參考；不能在本機重跑 |
| **B1. Height-smooth PPO seed42** | 512 env × 1000 iter × 24 step = **12,288,000** transitions（已全數完成）；`car_drift_height_smooth_task.json` | Isaac **13/17**，MuJoCo **11/17** | ✅ checkpoint `6ec411ef…` 與 metadata 一致 | ✅ `analyze_friction_grid --expected-summary` exit 0（本次）；task／catalog／trainer hash 與目前檔案一致（本次） | **主線參考基線候選** |
| B2. Height-smooth SAC seed42 | 同樣 12,288,000 transitions，23,981 次更新 | 0/17（兩引擎都是） | ✅ checkpoint `2722b411…` 一致 | ✅ 同上（本次） | 描述性比較；名義追蹤本身失敗 |
| B3. NMPC/CEM 開發版 | 未凍結；`qualified_nominal_pair_available=false` | 每個引擎 10/17，同 μ 兩引擎都通過 8/17 | 原生 50 Hz 函式庫不在本機（盤點 §4） | 本次未重算 | **不是基線**：名義配對未通過，沒有 freeze artifact |

### 2.2 補充分層 SA29（核心 [0, 3.14] s）

PPO 12/17（Isaac）、11/17（MuJoCo）；SAC 0/17、1/17；NMPC 每個引擎 11/17。PPO／SAC checkpoint 在本機，hash 與記錄一致（盤點 §4）。**SA29 的 NMPC 分析測試目前無法通過**（§5.3-c），所以這組 NMPC 結果的重現性目前未獲證明。

### 2.3 Hockey 支線（只列出，不判定是否納入）

| 基線 | 設定 | 結果 | 重現驗證 |
| --- | --- | --- | --- |
| v8（等值 μ DR，12,288,000 transitions） | PhysX 400 Hz／50 Hz，0–2.76 s，煞停 0.4–1.8 s | μ 0.7/0.7、0.9/0.7、0.9/0.9 各 1 次名義測試，11 項閘門全過；XY RMSE 2.72–2.77 cm | — |
| v9（split μ_s/μ_d = 0.9/0.7 DR） | 同上；由 v7 接續訓練 9,216,000 transitions | 0.9/0.7 下 **33/33**（1 名義 + 32 個預設初態擾動）；XY RMSE 2.21–3.03 cm | ✅ `audit_hockey_contact_model_framework.py` 重跑：`report.json`、`anchors.csv`、`signals.npz` **byte 相同**；16 項聚焦測試通過（本次） |

Hockey 的 11 項閘門寫死在 `scripts/evaluate_hockey_physx.py:387-395`：完整參考、進場速度 ≥0.8 m/s、接近段速度比 0.95–1.10、XY RMSE ≤0.15 m、peak ≤0.30 m、任一輪持續支撐、終端 ≥0.5 s 靜止、≥50 ms 受載側滑事件、車身航向 RMSE ≤15°、輪航向 RMSE ≤20°、傾角 <45°。另外還有雙輪幾何接觸（≤0.1 mm）的首要閘門。Hockey **沒有固定 μ 網格**，所以「XY 失敗 × μ」問題在這條線上目前沒有資料。

---

## 3. 資料切分

### 3.1 現況

| 項目 | 主線 | Hockey |
| --- | --- | --- |
| 參考軌跡 | 訓練與評估都用 traj4（SA29 另外訓練、另外評估） | 訓練與評估都用同一段 GMR s100 |
| 初始狀態 | 名義狀態，評估時關閉隨機事件 | 名義 + 32 個預設擾動（`config/hockey_bilateral_split_friction_v1.json` `evaluation_cases`） |
| 摩擦 | 訓練 DR 與評估網格分開（見 3.2） | 訓練 μ_d DR 0.3–1.2；評估 3 種材料 |
| Seed | 訓練 seed 42，只有一個 | 訓練 seed 42，只有一個 |
| Held-out | **沒有**；17 點網格已全部公開看過 | **沒有**；33 個案例已公開 |

所以目前**不存在** train／calibration／held-out 的三分切分。依研究定義 §6，現有網格只能當開發集或歷史驗證，不能重新命名成未見測試集。

### 3.2 訓練摩擦分佈與評估網格的關係（新發現，需讀回確認）

- **名目上**：robot static U[0.3,1.6]、dynamic U[0.3,1.2]（`config/evaluation_friction.json`），地面固定 1.0，combine rule 是 `average`（README §Experiment contract 第 4 點）。
- **評估時**：robot 和 ground 材料都設成 μ（PPO summary `isaac_material_readback`：robot 0.7、ground static/dynamic 0.7），所以有效 μ 就是網格值。
- **推導**：如果訓練真的是 average(robot, 1.0)，**有效**訓練範圍是 static [0.65,1.30]、dynamic [0.65,1.10]。

| 17 點網格分類 | 依名目範圍 | 依有效範圍（推導） |
| --- | --- | --- |
| 內插 | 0.30, 0.45, 0.60, 0.70, 0.80, 1.00, 1.20（7 點） | **0.70, 0.80, 1.00（3 點）** |
| 低側外插 | 0.01–0.20（6 點） | 0.01–0.60（9 點） |
| 高側外插 | 1.40–2.00（4 點） | 1.20–2.00（5 點） |

這會直接影響 RQ1 的解讀：低 μ 時 XY 失敗，可能其實是「訓練分佈外」造成，而不是摩擦本身的效應。**這一點是由 README 文字推導出來的，沒有讀回訓練期間實際的 combine 值**（待補項 G3）。

---

## 4. 參數設定（已核對值）

| 參數 | 值 | 依據 |
| --- | --- | --- |
| 主參考／補充 | `drift_drl_traj4`（motion `da9945c2…`、reference `afc22f19…`）／`doa_sa29_scale2` | `config/controller_references.json`（hash `b2c6062e…`，與 PPO metadata 一致） |
| 任務 profile | `config/car_drift_height_smooth_task.json`（`31de29a6…`）：sideslip 2.0、yaw_rate 1.0、speed 1.0、grounded 0.4、tilt −0.2；垂直加速度 −0.1（5 m/s²） | 檔案 hash 與 PPO/SAC metadata 一致 |
| 訓練預算 | 512 env、1000 iter、24 step/env → 12,288,000 transitions；seed 42 | `training_metadata.json` `training_budget` |
| 控制／記錄 | 50 Hz 控制；核心 2.56 s 有 513 個樣本 → 記錄頻率 200 Hz | summary `observed_core_samples` |
| 評估網格 | 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.70, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00（static = dynamic） | `config/evaluation_friction.json` |
| 名義 μ | 0.70 | README；NMPC 名義配對 |
| XY 篩選 | RMSE >0.15 m 或 peak >0.30 m（探索性） | `evaluation_friction.json`，**另外寫死在** `controller_metrics.py:173,183`（盤點 U6） |
| 腿部 PD | kp 500、kd 10（Isaac 讀回） | PPO summary `leg_pd_gain_readback` |
| Distinct μ_s/μ_d 配對 (1.4,1.0)、(1.6,1.2) | 在 `later_distinct_static_dynamic_pairs` 中，主線 evaluator **尚未支援** | `evaluation_friction.json`；README 第 692 行 |

---

## 5. 評估流程與可重現條件

### 5.1 評估流程（主線）

1. `materialize_controller_source.py` 建立隔離的全時程評估來源（μ 上限 2.0）。
2. `run_car_validation.py --full-horizon-diagnostic`：先檢查訓練完成狀態、材料 recipe、motion hash；在 Isaac 與 MuJoCo 對每個 μ 各跑一個確定性 episode；確認 Isaac policy-rate 觀測後才播放 MuJoCo；不覆寫既有網格。
3. `analyze_friction_grid.py` 由 `xy_trace.npz` 產生 `summary.json` 和 `topdown.png`；加上 `--expected-summary` 可以逐值比對。
4. 判定：依研究定義 §3 的 F/P/C/I。截尾不算通過；hash 不符屬於 I。

步驟 1–2 需要 Isaac Lab 2.3、授權 USD／MJCF、GPU。**本機只有 RTX 3060 Ti，沒有 Isaac Lab 環境**。歷史正式執行都在遠端主機（evidence 記錄 49 筆 `iv-ydqobidj40qbxys4nk24`）。所以**物理重跑目前只能在遠端做**，本機只能做步驟 3。

### 5.2 本次實際驗證

| 驗證 | 環境 | 結果 |
| --- | --- | --- |
| `analyze_saved_xy.py --trace results/car_baseline/xy_trace.npz` | archived venv | exit 0；`summary.json` dict 相等（PNG 的 bytes 不同，屬於繪圖後端差異，不影響指標） |
| `analyze_friction_grid.py --expected-summary`：PPO、SAC height-smooth | archived venv | 兩者 exit 0，17 個 case 都相等 |
| Hash 鎖：task profile、catalog、trainer、PPO/SAC checkpoint | — | 5/5 一致 |
| `audit_hockey_contact_model_framework.py` 重跑 | archived venv | 3 個資料輸出 byte 相同 |
| 主線指標測試 `test_xy_analysis.py`、`test_controller_metrics.py`、`test_physx_speed_reference.py` | archived venv | **73 passed** |
| 完整 `pytest tests`（485 檔） | latest venv（py3.12、numpy 2.5.4、mujoco 3.17、torch 2.15 cpu） | 8283 passed / 174 failed / 44 errors / 40 skipped |
| 完整 `pytest tests` | **archived venv**（py3.11.15、numpy 2.3.5、mujoco 3.8.1、torch 2.7.0 cpu、scipy 1.17.1） | **8345 passed / 107 failed / 38 errors / 6 skipped** |

archived venv 的版本取自歸檔 manifest 中最常出現的值（mujoco 3.8.1 出現 375 次、torch 2.7.0 出現 130 次、Python 3.11.15 出現 53 次、numpy 2.3.5 出現 17 次）。

### 5.3 剩下 145 個失敗／錯誤的分類（archived venv）

| # | 類別 | 約略數量 | 根因（已驗證） | 影響主線 XY？ |
| --- | --- | ---: | --- | --- |
| a | 依賴版本未鎖定 | latest venv 比 archived 多約 70 個 | `requirements.txt` 只寫 `>=`，而且**缺少 torch、opencv**（28 個測試檔需要 torch）；numpy 2.5 會造成最後一位小數的差異；未用 mujoco 3.8.1 時 native 測試會被跳過 | 否 |
| b | 授權資產無法讀取 | 32 | `/home/roy/tron1-learning-from-human/assets/robots/WF_TRON1A/mujoco/meshes/*.STL` 的擁有者是 root、權限 600 → MuJoCo 回報「STL is empty」（與盤點 U4 同源） | 否（影響物理／渲染測試） |
| c | **已提交的來源偏離實驗鎖定** | 約 50 | (1) `scripts/physx_mjlab_recipe.py` 在 `02766b8`（10/28，split friction）之後的 hash 從 `1365f4ee…` 變成 `804302b1…`，造成 v2f–v2o 的 protocol／continuation 測試和 `eval_ppo_backend_400hz` 失敗。(2) `scripts/run_nmpc_tracking.py` 有三個版本：SA29／friction development 鎖定 `5a5f714d…`、HEAD 是 `c6878e63…`、工作樹是 `bce84b92…`（CEM ground-contact v2b–v2o 鎖定的就是這個工作樹版本）。在 clean HEAD worktree 上，SA29 與 CEM v2d 測試同樣失敗 | **間接**：NMPC／SA29 歷史結果目前無法由 HEAD 或工作樹驗證 |
| d | CEM 15 項 stage reward 無法 bit-exact 重現 | 約 62 | `analyze_cem_ground_contact_v2b–v2m`、`render_cem_ground_contact_v2d`、`render_ppo_cem_matched_reward_400hz`；來源 hash 相符（使用工作樹版本）但數值不相等；原因**未查明**（可能是 CPU／BLAS／原生建置差異） | 否（CEM 接觸支線） |
| e | 錯誤訊息比對 | 2 | `test_physx_action_replay` 期待的例外訊息，被 (c) 的來源偏離例外搶先觸發 | 否 |

**重要**：工作樹未提交的 `run_nmpc_tracking.py`（+20 行 oracle 摩擦選項）正是 CEM v2 系列實驗實際使用的版本。如果用 `git checkout`／stash 丟棄它，這些結果就會失去可驗證的來源。

---

## 6. 進入實驗前仍需補齊的項目

### 6.1 需要使用者決策（不能由執行者自行決定）

| # | 決策 | 選項／建議 |
| --- | --- | --- |
| D1 | Hockey 支線是否納入本研究？ | 建議：若納入，需另立研究問題（煞停情境下的 μ 網格），不能和轉向 XY 合併 |
| D2 | 主線正式基線是否採用 height-smooth PPO seed42？ | 建議採用；SAC 當作對照組；NMPC 在 freeze 前排除 |
| D3 | 是否授權新增訓練 seed 或 held-out 條件？ | 單一 seed 只能做描述性結論；若要推論需要 ≥3 個 seed（研究定義 §5），並需要遠端 GPU 授權 |
| D4 | 是否提交工作樹的 7 個 M 檔與約 150 個未追蹤項？ | 建議先提交，至少要保留 `run_nmpc_tracking.py` 的 `bce84b92…` 版本 |

**決策記錄（2026-10-31，使用者）**
- D1：**hockey 煞停納入本研究**，定位為「宏觀展現滑動控制價值的動作」。研究定義 v1 排除 hockey 的部分因此需要修訂（新增 hockey 的研究問題，與轉向 XY 分開報告）。
- D3：**授權新增訓練 seed**。規格在 [`config/hockey_multiseed_replication_v1.json`](../config/hockey_multiseed_replication_v1.json)。已確認：完整 lineage 重跑、seed 43–47、v8 與 v9 兩組；滑動價值的對照組是**「不做特別動作、直接煞停」的不滑動對照組**（相同進場速度，seed 也用 43–47）。尚待處理：設計並凍結直接煞停的參考軌跡與 protocol，以及取得遠端上傳／啟動 payload 的核准。
- D2：**height-smooth PPO seed42 定為轉向主線的正式基線**。鎖定檔是 [`config/turn_formal_baseline_v1.json`](../config/turn_formal_baseline_v1.json)，內容包括 checkpoint 與來源雜湊、評估規則，以及名義 μ=0.70 下兩引擎的結果（Isaac 0.0565 m／MuJoCo 0.0498 m）。限制：只有單一 seed；17 點網格已經看過，屬於開發集；G1、G3 仍待完成。SAC 作為描述性對照組；NMPC 在 freeze 前排除。
- D4：**提交工作樹的修改**。提交範圍是本報告完成時已存在的 7 個修改檔與未追蹤項，以及本文件與相關設定。之後其他 session 新產生、仍在寫入中的 wheel_rig／sim2real／direction 系列檔案不在範圍內。

### 6.2 可重現性缺口

| # | 項目 | 完成條件 |
| --- | --- | --- |
| G1 | 鎖定環境：新增 `requirements-lock.txt`（python 3.11.15、numpy 2.3.5、scipy 1.17.1、mujoco 3.8.1、torch 2.7.0、mink 1.2.0、opencv） | 在全新 venv 中，完整測試結果不比 §5.2 的 archived venv 差 |
| G2 | 授權資產權限：讓目前使用者可以讀取 `WF_TRON1A/mujoco/**` | 類別 (b) 的 32 個測試通過；並比對 `export_model_sha256` |
| G3 | **讀回訓練期間實際的 robot/ground combine rule 與有效 μ** | 從訓練 manifest 或 PhysX 讀回確認 §3.2 的推導；必要時重新標記網格點屬於內插或外插 |
| G4 | 來源偏離：為 `physx_mjlab_recipe.py`、`run_nmpc_tracking.py` 的歷史版本建立 source snapshot，或讓測試改為比對 snapshot | 類別 (c) 的測試全部通過，而且不修改歷史 hash 鎖 |
| G5 | 查明 CEM reward bit-exact 失敗的原因 | 類別 (d) 通過，或者文件說明它是平台相依並改用容差（需使用者同意） |
| G6 | 遠端執行環境的記錄（主機、Isaac Lab 版本、GPU、driver） | 每次物理實驗的 manifest 都含這些欄位 |

### 6.3 方法缺口（沿用研究定義 §8，本次狀態更新）

| # | 項目 | 本次狀態 |
| --- | --- | --- |
| M1 | `analyze_saved_xy.path_error` 沒有檢查核心起點與丟幀 | 仍未實作；在新 preflight 加入之前，不改寫既有結果 |
| M2 | XY 閾值有兩個來源（config 與程式內寫死） | 目前兩者一致；應改成只從 config 讀取 |
| M3 | 資料切分 manifest（train／calibration／held-out 與 hash） | 不存在；取決於 D3 |
| M4 | 高角度分層門檻 A_high 的校準集 | 不存在；目前只能做連續角度描述 |
| M5 | Distinct μ_s/μ_d 在主線 evaluator 的支援 | 未支援（hockey 線已支援） |
| M6 | 研究定義指出的 pytest 未執行問題 | **已解決**：主線指標測試 73/73 通過 |

### 6.4 進入實驗的門檻

只要 D1、D2 已決定，且 G1、G3 完成，就可以開始**描述性**的 RQ1/RQ2 分析（使用現有網格，不需要新的物理模擬）。如果要做**推論性**實驗或新網格，還必須先完成 D3、M3，並取得遠端 GPU 授權。G2、G4、G5 影響的是支線與歷史結果的驗證，不阻擋主線描述性分析。

---

## 7. 驗收對照

| 任務要求 | 段落 | 狀態 |
| --- | --- | --- |
| 確認現有基線 | §2 | 完成：主線 5 項、SA29、hockey 2 項，各附判定與重現證據 |
| 資料切分 | §3 | 完成：確認沒有切分，並新發現有效 μ 範圍的問題 |
| 參數設定 | §4 | 完成，核心值都附 hash 或讀回依據 |
| 評估流程 | §5.1 | 完成 |
| 可重現條件 | §5.2–5.3 | 完成：三種環境實測，失敗依根因分類 |
| 進入實驗前待補項 | §6 | 決策 4 項、可重現性 6 項、方法 6 項、門檻 1 段 |
