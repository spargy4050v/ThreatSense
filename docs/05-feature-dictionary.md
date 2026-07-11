# Feature dictionary

## How to read the names

The dataset uses `plugin.measure` names. The prefix identifies the memory-
forensics view and the suffix identifies a count or average.

| Prefix | General meaning |
| --- | --- |
| `pslist` | Process-list statistics from active process structures. |
| `dlllist` | Dynamic-link libraries observed in process address spaces. |
| `handles` | Windows kernel-object handle counts. |
| `ldrmodules` | Consistency of modules across loader-maintained lists. |
| `malfind` | Indicators of suspicious injected or executable memory regions. |
| `psxview` | Cross-view process visibility and hidden-process discrepancies. |
| `modules` | Loaded kernel module counts. |
| `svcscan` | Windows service and driver scan statistics. |
| `callbacks` | Registered kernel callback statistics. |

Terms such as `not_in_*` describe disagreement between forensic views. Such
disagreement can be caused by legitimate edge cases, rootkit-like hiding,
unlinked modules, corruption, timing, or collection artifacts; it is not proof
of malware by itself.

## Source columns not used as numeric inputs

| Column | Treatment |
| --- | --- |
| `Category` | String sample/type/family identifier. Removed by numeric selection and retained as random-row context. |
| `Class` | Target label. Mapped to `0/1`; never used as an input feature. |

## Retained model features

These 31 columns are stored in `models/features.json` in the exact order
expected by the scaler, model, and Flask API.

| Feature | Meaning in the project |
| --- | --- |
| `pslist.nproc` | Number of listed processes. |
| `pslist.nppid` | Parent-process identifier/count statistic from the process list. |
| `pslist.nprocs64bit` | Count of 64-bit processes. |
| `handles.nport` | Count of port handles. |
| `handles.ndesktop` | Count of desktop handles. |
| `handles.ndirectory` | Count of directory-object handles. |
| `ldrmodules.not_in_init_avg` | Average module inconsistency against the initialization-order list. |
| `malfind.ninjections` | Number of suspicious memory-injection findings. |
| `malfind.commitCharge` | Committed-memory amount associated with suspicious regions. |
| `malfind.protection` | Encoded protection statistics for suspicious memory regions. |
| `malfind.uniqueInjections` | Count/average of unique suspicious injections. |
| `psxview.not_in_pslist` | Processes visible elsewhere but missing from `pslist`. |
| `psxview.not_in_eprocess_pool` | Processes missing from the EPROCESS pool view. |
| `psxview.not_in_ethread_pool` | Processes missing from the ETHREAD pool view. |
| `psxview.not_in_pspcid_list` | Processes missing from the process/client-ID view. |
| `psxview.not_in_csrss_handles` | Processes missing from CSRSS handle visibility. |
| `psxview.not_in_session` | Processes missing from the session view. |
| `psxview.not_in_deskthrd` | Processes missing from desktop-thread visibility. |
| `psxview.not_in_pslist_false_avg` | Average false/disagreement statistic for `pslist`. |
| `psxview.not_in_eprocess_pool_false_avg` | Average false/disagreement statistic for EPROCESS pool. |
| `psxview.not_in_ethread_pool_false_avg` | Average false/disagreement statistic for ETHREAD pool. |
| `psxview.not_in_pspcid_list_false_avg` | Average false/disagreement statistic for process/client IDs. |
| `psxview.not_in_csrss_handles_false_avg` | Average false/disagreement statistic for CSRSS handles. |
| `psxview.not_in_session_false_avg` | Average false/disagreement statistic for sessions. |
| `psxview.not_in_deskthrd_false_avg` | Average false/disagreement statistic for desktop threads. |
| `modules.nmodules` | Count of loaded kernel modules. |
| `svcscan.fs_drivers` | Count of file-system driver services. |
| `svcscan.interactive_process_services` | Count of interactive-process services. |
| `callbacks.ncallbacks` | Total registered callback count. |
| `callbacks.nanonymous` | Count of callbacks without a resolved module/name. |
| `callbacks.ngeneric` | Count of generic callback entries. |

## Flagged and removed features

These 24 features achieved mean single-feature cross-validation accuracy at or
above 0.85 on the seed-42 training split.

| Feature | Meaning and review concern |
| --- | --- |
| `svcscan.shared_process_services` | Shared-process service count; plausible system-state signal but may encode VM configuration. |
| `svcscan.nservices` | Total service count; broad and potentially environment-specific. |
| `svcscan.kernel_drivers` | Kernel-driver service count; security-relevant but also machine-configuration dependent. |
| `handles.nmutant` | Mutex/mutant handle count; malware may use mutexes, but application mix also affects it. |
| `dlllist.avg_dlls_per_proc` | Average libraries per process; plausible runtime-complexity signal. |
| `handles.nevent` | Event-object handle count; broad runtime activity signal. |
| `handles.avg_handles_per_proc` | Average handle count per process; broad workload/system-state signal. |
| `handles.nsection` | Section-object handle count; relevant to mapped memory and code, but nonspecific. |
| `pslist.avg_handlers` | Average handles per listed process. |
| `handles.nhandles` | Total handles; strongly influenced by machine workload and capture conditions. |
| `handles.nkey` | Registry-key handle count; plausible malware activity and normal workload signal. |
| `dlllist.ndlls` | Total observed DLL count. |
| `handles.nsemaphore` | Semaphore handle count. |
| `handles.nthread` | Thread handle count. |
| `ldrmodules.not_in_load` | Modules absent from loader load-order visibility; plausible hidden/unlinked-module indicator. |
| `ldrmodules.not_in_mem` | Modules absent from memory-order visibility; plausible hidden/unlinked-module indicator. |
| `handles.ntimer` | Timer handle count. |
| `ldrmodules.not_in_init` | Modules absent from initialization-order visibility. |
| `svcscan.nactive` | Active service count; broad machine-state signal. |
| `handles.nfile` | File handle count; plausible malicious file activity but also workload dependent. |
| `pslist.avg_threads` | Average threads per process. |
| `svcscan.process_services` | Process-hosted service count. |
| `ldrmodules.not_in_load_avg` | Average load-order inconsistency. |
| `ldrmodules.not_in_mem_avg` | Average memory-order inconsistency. |

## Manual review conclusion

No flagged name is an obvious target label, row identifier, family string, or
post-outcome field. Several are plausible behavioral indicators. The broadest
aggregate features - especially total service, active service, total handle,
average handle, and general workload counts - are the best candidates for
collection-environment sensitivity.

The ablation showed that removing all 24 lowered accuracy by 0.802 percentage
points. Therefore the screen should not be described as having discovered
confirmed leakage. A stronger next step would test selected groups across a
different collection environment or a group-aware split.

## Why feature order matters

The scaler stores statistics positionally, and the MLP's first layer expects 31
ordered values. `models/features.json` is the schema contract. Reordering
values, omitting a feature without filling it, or using a different scaler can
produce invalid predictions even when array dimensions still match.
