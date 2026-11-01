# TRON1 滑移角度與 XY 失敗研究定義 v1

日期：2026-10-31。任務：t_16c65cbedb15。狀態：草案已持久化，任務維持 progress。
本文件定義分析契約，不是基線已確認或實驗已完成的證明。本次僅查閱現存檔案、嘗試既有離線測試及文件靜態驗證；未訓練、模擬、調參或變更研究範圍。

## 1. 範圍與局部盤點

研究對象是既定 TRON1 平面轉向中的行進／輪平面角度、固定摩擦條件下的 XY 追蹤失敗，以及 Isaac/PhysX 與 MuJoCo 差異。人體工作仍暫停；不納入新的 hockey 任務、不宣稱高速能力或真實輪胎校準。高度、離地、姿態與角度本身均不是 XY 失敗門檻。

| 依據（相對專案根目錄） | 已核對用途 |
| --- | --- |
| README.md 開頭、研究流程第 5–6 點 | 主問題、速度保持、全核心判定與探索性限制；README 有歷史時態，不能單靠文字宣告目前基線合格 |
| config/controller_references.json | 主參考／補充參考、座標單位、來源與 motion/reference 雜湊 |
| config/evaluation_friction.json | 固定摩擦網格、XY 閾值及原始核心區間 |
| scripts/controller_metrics.py：heading_signals、summarize_turn | 角度週期、速度有效性、接觸 gate、速度／XY 公式 |
| scripts/analyze_saved_xy.py：observed_episode、path_error、pair_error、summarize | 原始 episode prefix、時間內插、截尾與歷史 XY 指標 |
| upstream/tron1-learning-from-human/training/tron1_contact_slip.py | 真實接觸點相對速度與法向力加權橫向滑移 |
| config/nmpc_friction_development.json | 開發資料的控制器／來源雜湊；明載 qualified_nominal_pair_available=false、formal_controller_freeze_claimed=false |
| results/car_baseline/xy_trace.npz、xy_trace.provenance.json、summary.json | 可取得的歷史三個摩擦配對之緊湊軌跡與結果；不是完整新基線 |
| tests/test_xy_analysis.py、tests/test_controller_metrics.py | 既有離線指標測試，見第 9 節執行限制 |

主參考 drift_drl_traj4：核心 [1.44,4.00] s；補充 doa_sa29_scale2：核心 [0.00,3.14] s，描述性窗口 [0.78,1.54]、[1.88,3.04] s。分層各報，禁止合併成功率。合成恢復段與靜止尾端不屬核心。兩者源自 CARLA、時鐘為重建估計，不能稱真人／實車捕捉。

前置「程式／資料盤點」任務 t_980998173fdb 在本次查詢仍為 todo，專案 Library 為空。本節只完成本文件所需局部盤點，未冒稱全專案盤點完成。HEAD 與讀取來源的實際 SHA-256 見同名 evidence.json；工作樹已有大量修改，HEAD 不能單獨代表來源版本。

## 2. 滑移角度的操作化定義

採世界 Z-up，位置 m、線速度 m/s、角速度 rad/s、時間 s；姿態為 world-from-link 的 wxyz 單位四元數。XY 位置使用 root/base link 原點，速度使用 base_Link COM；不得把 COM 位置與 link 原點參考混用。左右輪固定順序 L/R。

令 base COM 水平速度 v=(vx,vy)，s=norm(v)，course ψv=atan2(vy,vx)。車身前向為 Rbase*(1,0,0)，其 XY 方位 ψb。輪軸 a_j=Rwheel_j*(0,1,0)，輪滾動平面方向 r_j=a_j×(0,0,1)，方位 ψw_j=atan2(r_y,r_x)。

- 車身側滑角 β=atan2(sin(ψv−ψb),cos(ψv−ψb))，有向角約 [-180°,180°]。
- 主要輪平面幾何夾角 γ_j=0.5*atan2(sin(2*(ψv−ψw_j)),cos(2*(ψv−ψw_j)))，軸向角約 [-90°,90°]，正負表示世界 XY 中逆／順時針相對差。
- 輪心版本 γwheel_j 使用該輪心 XY 速度 course，獨立命名，不可與 base COM 版本混合。
- 計算內部 rad，報表 deg。相應速度 >=0.25 m/s 才有效；投影軸長 >1e-6 才有效。輪心版本使用輪心速度 gate；base-to-wheel 使用 base 速度 gate。無效樣本標記 null/mask，不算零角度。
- 報左右輪 mean(abs γ)、P95(abs γ) 與軸向圓平均 0.5*arg(mean(exp(2iγ)))；β 用 arg(mean(exp(iβ)))。合向量長 <=1e-10 時圓平均為 null。必報有效樣本數／核心樣本數。
- 輪自轉與四元數正負不改變輪平面角。catalog 的 signed wrap 描述未明確說明軸向週期，本研究以 controller_metrics.py 的 axial 實作為明確操作規格；舊 signed-wrap 統計須標記版本後重算，不能直接混用。

「高角度」不增設成功或失敗 gate。主要以連續 |γ|（deg）描述現有固定核心。若後續需要高／低角度分層，事先在校準集鎖定 |γ| 的第 75 百分位 A_high，固定分位數算法與左右輪規則，以 >=A_high 分組；測試資料不得重估。尚無獨立校準集，此分層目前不可宣稱已凍結。

接觸滑移是另一種量：每個真實接觸點 p，u=vCOM+ω×(p−pCOM)−vground；單位接觸法線 n，橫向單位向量 l=(a−(a·n)n)/norm(a−(a·n)n)，vlat=u·l，單位 m/s。同輪各有效接觸點以正法向力 F 加權，slip_j=sqrt(sum(F*vlat²)/sum(F))。軸投影退化時無效。每時刻 ground-only 接觸 count>0 且世界向上淨力 Fz>5 N 才納入；核心彙整為 sqrt(mean(slip_j²))，分母是該輪有效接觸樣本數。無接觸／缺感測為 null，不是零滑移。參考幾何沒有接觸資料，不得宣稱接地滑移角。這是橫向速度，並非完整輪胎 slip ratio。

## 3. XY 失敗事件、時間與分母

一個分析單位為唯一 episode，鍵至少包含 reference_id、task_profile hash、controller/checkpoint hash、engine/version、μstatic、μdynamic、training_seed、evaluation_seed、initial_state hash、episode_id。重複匯出或分析 cell 不能增加樣本數。

令 t_i 是原生物理記錄時間，p_i 是 root XY，p_ref(t_i) 為時間對齊參考的逐軸線性內插，e_i=norm(p_i−p_ref(t_i))（m）。禁止最佳事後剛體對齊、時間扭曲、按失敗時間裁窗口、外插或拼接 reset 後軌跡。只允許事先由座標契約指定的初始座標轉換並保存其參數。

核心集合 C={i:a−1e-9<=t_i<=b+1e-9}，N=|C|。必須覆蓋 a 至 b、N>=2、時間嚴格遞增、位置有限、無 reset，且原生記錄間距符合宣告記錄頻率；最大允許間距為 1.5 倍宣告週期（時間數值容差 1e-9 s）。丟幀資料另列資料品質不合格，不以內插掩蓋。
完整核心 RMSE=sqrt(sum_C e_i²/N)，peak=max_C e_i。主要分析沿用等間距原生樣本平均；異常不等間距不能直接混用。

完整 episode 的探索性 XY 失敗 F=(RMSE>0.15 m OR peak>0.30 m)。等於閾值不算超限；用未四捨五入數值判斷。這兩值沿用 config，不是物理輪胎閾值或新校準結果。完整且未超限為 P；未完整為 C（截尾，正式 verdict=null），另外保存已觀察峰值。啟動失敗、缺檔／hash 不符／記錄損壞為 I（無有效判定），不得當成物理 XY 失敗或成功。

描述性事件時間 t_peak_cross=min{t_i in C:e_i>0.30 m}（s），單 episode 最多計一次；未觀察跨越時 null。持續時間可報超限樣本數/N（僅完整等間距核心）。RMSE 是整段統計，不可把「RMSE 首次失敗時間」等同峰值跨越。若有跨越，發生時刻只能定位於相鄰原生樣本間；核心第一筆已超限時註記核心入口前起始未知。

每個固定 reference/controller/engine/μ 分層：
- 計畫 episode 數 Nplanned=F+P+C+I（包含未啟動者）；完整可判定數 Ncomplete=F+P。
- 完整條件失敗率 F/Ncomplete；Ncomplete=0 時 null。必附 C、I 與覆蓋率 Ncomplete/Nplanned。
- 全計畫失敗比例的保守未知界 [F/Nplanned,(F+C+I)/Nplanned]，不是信賴區間。C 即使已見局部超限仍依契約保持截尾，但可另報已觀察跨越。
- 17 點網格的通過點數/17 僅是固定網格覆蓋描述，不是隨機母體成功機率。多個 μ、時間樣本、左右輪均不是獨立重複實驗。

實作差距：analyze_saved_xy.path_error 的 core_complete 只檢查終點，未驗證起點完整性；pair_error 也須外部檢查共同核心起點。controller_metrics.summarize_turn 已檢查兩端與樣本數，但未檢查丟幀。後續使用前需在 adapter/preflight 加入上述檢核與測試；本次未修改程式或重新標記既有結果。

## 4. 可測試問題與假設

| 問題／假設（待驗證，非結論） | 比較與可反證結果 |
| --- | --- |
| RQ1：固定控制器的 XY 失敗如何隨 μ 改變？H1：低於名義 μ=0.70 的條件較名義值有較高 RMSE／失敗比例 | 同參考、同 checkpoint、同 seed/初態配對，各 μ 對 0.70 報 ΔRMSE、Δpeak、ΔF；若未增加則不支持。全網格保留，不能假設單調 |
| RQ2：相同數值 μ 的兩引擎是否給出不同 XY 表現？H2：配對 ΔRMSE 非零且部分 verdict 不一致 | 每 μ 報 MuJoCo−Isaac 差、verdict 交叉表、共同核心路徑差；缺任一完整核心即不形成完整配對。不是摩擦物理等效測試 |
| RQ3：較大幾何角度是否伴隨較大追蹤誤差／接觸滑移？H3：固定窗口內 |γ| 與 e、接觸 slip 正向相關 | 僅探索性、左右輪分開、按 episode 算相關再彙整，控制速度、μ、參考時間／曲率及接觸覆蓋；常數或不足樣本為 null。相反／零關聯不支持。不得由同時相關推論角度造成失敗 |
| RQ4：既有 PPO、SAC、固定 NMPC 的 XY 與速度保持是否有取捨？H4：較低 RMSE 不必然伴隨較高速度保持 | 使用已選定版本配對報 ΔRMSE、Δ速度保持、接觸滑移、負載／延遲，不以單一總分排名。保留名義失敗；此假設可被一致改善的資料反駁 |

RQ1 為主要問題；其餘為次要／探索性。現有單一主軌跡無法識別「改變滑移角本身」的因果效應，不能以分幀增加獨立 n。無新軌跡或干預的授權，本次不提出範圍外實驗。

## 5. 指標、比較基線與不確定性

主要終點是完整核心 reference XY RMSE（m）；peak（m）與探索性二元失敗共同呈現。下列量均獨立呈現，不改寫 XY verdict：

| 指標 | 計算／分母及限制 |
| --- | --- |
| 速度保持 | mean(s_actual)/mean(s_ref)，無單位，可 >1；參考均速 <=1e-12 m/s 時 null。另報 mean(s_actual) m/s 及 count(s_actual<0.8*s_ref)/N |
| 轉向幅度 | 有效 course 連續 unwrap 後末−首（deg）；全窗口 course 有效才報，不跨低速空洞 |
| 角度與接觸 | 第 2 節各輪角度、有效率、接觸樣本數/N 與接觸 slip RMS；接觸條件選樣可能產生偏差，須同報全窗口幾何值 |
| sim2sim 路徑差 | 在 Isaac 原生時間上內插 MuJoCo，sqrt(mean(norm(pI−pM)²))（m）；僅完整共同核心稱 full_core；不能取代各自對參考誤差 |
| 計算負擔 | 每控制步 wall-clock 耗時的 median/P95/max（ms），超宣告控制週期的步數/全部計時步數；分開 warmup 與 steady-state，記硬體／同步方法 |
| 致動負擔 | 每關節實測 torque RMS（Nm）、peak abs（Nm），速度 peak（rad/s），飽和樣本數/有效 joint-time 樣本數；須記實際動態限值與 clipping 判準，資料不足為 null |

基線分三層：(A) car_baseline 三配對僅用於分析重算相容性；(B) 現存 height-smooth PPO/SAC/NMPC 開發版本作描述性比較，明列 task/observation/compute 不同；(C) 尚待下一任務確認的正式凍結基線。不得以 B 取代 C。固定 μ 網格沿用 evaluation_friction.json 的 17 點；不啟動新網格。distinct static/dynamic 的支援須另外核對，目前不默認可執行。已知 μ oracle 僅可單列特權診斷，不能和不知道 plant μ 的控制器當同等資訊比較。

主要報每 episode 原始點、配對差與各分層樣本數。單一訓練 seed／確定性 episode 不產生跨 seed 推論 CI，不能 bootstrap 幀或 μ 點冒充重複。若未來取得足夠獨立訓練 seed，先鎖定數量與停止規則，再按 training_seed 整群重抽（保留全部 μ／引擎配對；seed 20261031、10000 次、百分位 2.5/97.5）報平均配對差 95% CI；樣本過少時只報全點和範圍並說明不足。此為分析規則，不是本次執行授權。多重次要比較全部揭露，不擇顯著 μ；若轉為正式假設檢定，需事先鎖定比較家族並用 Holm 校正。

摩擦「邊界」只能報相鄰已觀察 μ 的 verdict 改變區間；網格可能非單調，列出所有改變，不任選單一臨界 μ，不外插。

## 6. 防資料洩漏與混淆控制

先建立資料 lineage manifest，以原始來源 motion／episode／checkpoint／訓練 seed 分組；相同 episode 的裁切、重新匯出、相鄰幀及同源軌跡不跨資料分割。既有公開／已看過的 μ 網格一律開發或歷史驗證，不能重新命名為未見測試集。

基線確認階段須明確鎖定 train、nominal calibration、held-out evaluation 清單與 hashes。僅 calibration 可決定模型／閾值／分層；測試解封前鎖定控制器、normalization、reward、source、參考、分析規則。未證明未使用的 holdout 就只能描述性比較，不宣稱泛化。nominal-qualified freeze 仍須原工作流程的兩引擎名義配對證據，不能因本定義而放寬。名義失敗照報，不再搜尋調參。

| 混淆來源 | 控制與事先指定敏感度 |
| --- | --- |
| 軌跡、速度、曲率、空間／時間縮放 | 主／補充分層；固定來源 hashes；保留速度與 course 指標，不在兩軌跡間歸因控制器差異 |
| XY 閾值任意性 | 主規則不變；敏感度 RMSE {0.10,0.15,0.20} m × peak {0.20,0.30,0.40} m 全部報，不挑有利組合 |
| 低速角度不穩定與接觸 gate | 速度 gate {0.10,0.25,0.50} m/s；向上力 gate {1,5,10} N，報有效覆蓋，不把缺失補零 |
| 控制／物理頻率與記錄降採樣 | 記錄所有 dt、decimation；原生均勻時鐘為主；若另報時間積分 RMS（梯形積分除窗口長度）則獨立命名，不改寫歷史 sample RMS |
| 引擎接觸模型 | 固定／讀回材料 combine rule、摩擦、restitution、solver、substeps、接觸幾何、normal compliance；相同 μ 不表示同一輪胎定律 |
| 訓練 recipe、資訊與算力 | 記錄 reward/profile、觀測、randomization、interaction budget、optimizer reuse、硬體、MPC 預測模型、oracle 資訊；同互動步數不等於同計算量 |
| early termination／姿態喪失 | C/I 分列與未知界；新全時程與舊提前停止分層；高度／姿態僅診斷，不增加 XY 失敗 gate |
| 選樣與重複使用 | 各唯一 episode 計一次；左右輪、μ、幀不是獨立重複；禁止看完測試選窗口或控制器 |

上述敏感度僅為預先規格，尚未執行；不改動現有設定。

## 7. 可重現性與資料契約

每 episode 需有：唯一鍵與資料分割、parent episode/reference lineage、checkpoint/normalization/task hashes、source commit 與 dirty files hashes、完整 CLI/config/env、Python/依賴及引擎版本、硬體、所有 seed、initial state、control/physics/log dt、材料設定及 runtime readback、termination/reset/valid_mask、原始輸出 hashes。

最低 XY 資料：time_s[T]、root_position_world_m[T,3]、reference_time_s[R]、reference_root_position_world_m[R,3]。角度另需 root COM velocity[T,3]、root quaternion[T,4]、wheel quaternion[T,2,4]、wheel-center velocity[T,2,3]。速度保持需 reference COM XY speed[R]。接觸另需 ground-only counts[T,2]、world force[T,2,3]、slip RMS[T,2] 及其生產器 hash；若重算需接觸點／法線／正法向力、body COM 位姿速度、ground velocity。接觸力的瞬時／平均／impulse-to-force 定義與時鐘必須紀錄。感測器飽和／最大接觸數需檢查。負載與耗時需原始 torque、速度、限值及每步 wall time。

預檢：hash identity、shape/單位/LR 名稱、四元數 norm、時間單調且覆蓋、reset prefix、參考核心匹配、原生頻率／缺值、接觸來源、唯一 episode 去重。raw 保持唯讀；衍生資料在新目錄，記錄規格版本與生成命令。禁止默認接受 hash 不符的重建 traj4；數值近似不等於 byte lock 通過。授權資產與 checkpoint 可用性須逐檔核對，不從 README 推定已存在。

最低輸出每列：唯一鍵、來源 hashes、window、Nplanned/Ncomplete、coverage、status F/P/C/I、RMSE/peak/verdict、角度 mask 分母、接觸分母、速度比、缺失原因；另保留配對 ID、各 μ 表和等比例俯視 XY 圖，全部 nominal failures 也包含。

## 8. 未決事項與階段門檻

1. 前置全專案程式／資料盤點尚未完成；本文件未盤點全部 results 或全部 checkpoint、授權資產。
2. 正式 baseline 的確切 checkpoint/source/profile、兩引擎名義配對、資料切分及未見 holdout manifest 尚未在本次逐一驗證。既有開發證據不能補作未見測試。
3. 環境缺 pytest，既有指標測試未成功執行；需使用已核准且依賴完整的環境重跑第 9 節命令。
4. 起點覆蓋／丟幀 preflight 差距尚未實作驗證。接觸時鐘／力定義、各候選資料的 sensor completeness 尚待逐組核對。
5. 高角度二分閾值的校準資料尚未凍結；目前連續角度描述可用。獨立 seed 數與 inferential sample size 未確定，目前只允許描述性結論。

這些是具體缺口，不以本文件將任務改為 review/done。先完成盤點及必要驗證，再由負責者依本規格確認基線；本次不進入實驗。

## 9. 驗證記錄與驗收對照

本次成功讀取上述程式／設定／既有測試；evidence.json 記錄當下來源 bytes 的 SHA-256 與文件結構／公式關鍵項檢查結果。來源文字核對不是執行測試通過。

嘗試命令：
- PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_xy_analysis.py tests/test_controller_metrics.py → exit 127，python 不存在。
- PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_xy_analysis.py tests/test_controller_metrics.py → exit 1，No module named pytest。

既有測試涵蓋：歷史 trace 重算、拒絕外插、截尾非通過；有向／軸向角、輪自轉不變、低速 mask、接觸力 gate、嚴格閾值、忽略 Z 作失敗條件、缺接觸非零。此列是閱讀到的測試內容，並非本次執行通過數。

| 驗收項 | 文件段落 | 結果 |
| --- | --- | --- |
| 滑移角操作化與單位 | §2 | 定義完整，附有效性與接觸區別 |
| XY 事件／閾值／分母 | §3 | 定義完整，揭露既有實作差距 |
| 可測試問題與假設 | §4 | 四項；因果與泛化限制明列 |
| 混淆／敏感度／防洩漏 | §6 | 規則完整，實際 split 尚缺 |
| 基線／指標／不確定性 | §5 | 方法已定義，正式基線未確認 |
| 重現性與資料 | §7–8 | 資料契約完整，實際全盤點未完成 |
| 實際檢查與成果 | §9、evidence.json | 靜態驗證可執行，pytest 未通過執行前提 |

本規格是後續補齊證據的持久化成果；不宣稱整個研究定義任務已驗收通過。
