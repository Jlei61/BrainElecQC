"""Deterministic internal QC ZIP, with member hashes and independent readback.

Source: Cortrix (VISVB) SEEG quality-control implementation.
"""
import csv
import hashlib
import html
import io
import json
import zipfile

from ..storage import canonical_bytes

EXPORT_FORMAT = "brainelecqc_seeg_qc_zip_v1"

_EVENT_FIELDS = ["event_id", "channel_id", "stream_id", "segment_id", "start_sample", "stop_sample",
                 "recording_start_seconds", "recording_stop_seconds", "duration_seconds", "recovery_status",
                 "termination_reason", "score_effect", "reasons", "stages", "features"]


def _csv(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        values = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            # Prevent formula execution when reviewers open CSV in a spreadsheet.
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                value = "'" + value
            values[key] = value
        writer.writerow(values)
    return output.getvalue().encode("utf-8-sig")


def build_package(snapshot):
    summary = snapshot["review"]["summary"]
    report = snapshot["report"]
    files = {"qc_result.json": canonical_bytes(snapshot),
             "channel_map.json": canonical_bytes(report["channel_map"]),
             "channels.csv": _csv(summary["channels"], ["channel_id", "raw_name", "stream_id", "total_seconds",
                                                        "kept_seconds", "excluded_seconds", "unknown_seconds", "reviewed_ratio"]),
             "decisions.csv": _csv(snapshot["review"]["decisions"], ["decision_id", "channel_id", "stream_id", "scope",
                                                                       "segment_id", "start_sample", "stop_sample", "action",
                                                                       "response", "reason", "reviewer", "reviewed_at", "evidence_window_ids"]),
             "windows.csv": _csv(report["windows"], ["window_id", "channel_id", "stream_id", "segment_id", "start_sample",
                                                       "stop_sample", "duration_seconds", "score", "flags", "metrics"])}
    esc = lambda value: html.escape(str(value))
    rows = "".join("<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key in
                                    ("raw_name", "kept_seconds", "excluded_seconds", "unknown_seconds")) + "</tr>"
                   for row in summary["channels"])
    body = ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>SEEG 基础 QC</title>"
            "<style>body{font:16px system-ui;margin:40px;max-width:1100px}table{border-collapse:collapse}"
            "td,th{padding:8px;border:1px solid #bbb}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            f"<h1>SEEG 基础技术 QC</h1><p>记录：{esc(snapshot['recording_id'])} · 版本：{esc(snapshot['run_id'])}</p>"
            f"<p>初评分：{esc(report['initial_score'].get('total'))}；复核后评分：{esc(summary['score'])}；"
            f"审核覆盖：{summary['reviewed_ratio']:.1%}；待定：{summary['unknown_ratio']:.1%}</p>"
            f"<p>自动扫描覆盖：{report['coverage']['scanned_fraction']:.1%}；自动规则：{esc(report['profile'])}</p>"
            f"<p>已复核部分保留率：{esc(summary['retained_percent'])}%；候选证据已裁决：{summary['evidence_reviewed_ratio']:.1%}。</p>"
            "<p>技术复评分只按明确的候选裁决更新，待定部分沿用初评风险；保留率单独统计。未复核数据不视为合格。</p>"
            f"<p>评价：{esc(snapshot['assessment']['status'])} · 范围：基础信号技术质量</p>"
            "<table><thead><tr><th>通道</th><th>确认可用秒数</th><th>排除秒数</th><th>待定秒数</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><h2>限制</h2><pre>{esc(json.dumps(snapshot['assessment']['limitations'], ensure_ascii=False, indent=2))}</pre>"
            f"<h2>复核记录</h2><pre>{esc(json.dumps(snapshot['review']['decisions'], ensure_ascii=False, indent=2))}</pre></html>")
    files["report.html"] = body.encode("utf-8")
    if "waveform_events" in report:
        events = report["waveform_events"]
        files["waveform_events.csv"] = _csv(events, _EVENT_FIELDS)
        files["windows.csv"] = _csv(report["windows"], ["window_id", "channel_id", "stream_id", "segment_id",
                                                      "start_sample", "stop_sample", "duration_seconds", "score",
                                                      "flags", "metrics", "waveform_event_ids"])
        names = {row["channel_id"]: row["raw_name"] for row in report["channel_map"]}
        kinds = {"jump": "跳变", "drift": "漂移", "plateau": "平台", "recovery": "恢复确认"}
        event_rows = []
        for event in events:
            stages = "<br>".join(f"{esc(kinds.get(stage['kind'], stage['kind']))}："
                                  f"[{stage['start_sample']}, {stage['stop_sample']})"
                                  for stage in event["stages"])
            status = "已确认恢复" if event["recovery_status"] == "confirmed" else "截至此处未确认恢复"
            event_rows.append(f"<tr><td style='overflow-wrap:anywhere'>{esc(event['event_id'])}</td>"
                              f"<td>{esc(names.get(event['channel_id'], event['channel_id']))}</td>"
                              f"<td>{event['recording_start_seconds']:.4f}</td><td>{event['recording_stop_seconds']:.4f}</td>"
                              f"<td>{status}</td><td>{stages}</td></tr>")
        event_html = (f"<h2>波形异常候选（{len(events)}）</h2><p>跳变、漂移、平台及恢复阶段；区间为半开区间，候选供人工复核。</p>"
                      "<table><thead><tr><th>事件</th><th>通道</th><th>开始秒</th><th>结束秒</th><th>恢复状态</th><th>阶段</th>"
                      f"</tr></thead><tbody>{''.join(event_rows)}</tbody></table>")
        files["report.html"] = body.replace("</html>", event_html + "</html>").encode("utf-8")
    manifest = {"schema_version": 1, "format": EXPORT_FORMAT, "scope": snapshot["scope"],
                "run_id": snapshot["run_id"], "recording_id": snapshot["recording_id"],
                "input_fingerprint": snapshot["provenance"]["content_sha256"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["manifest.json"] = canonical_bytes(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    verify_package(payload, snapshot)
    return payload


def verify_package(payload, snapshot):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if set(archive.namelist()) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("QC export member set mismatch")
        for name, checksum in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"QC export checksum mismatch: {name}")
        if json.loads(archive.read("qc_result.json")) != snapshot:
            raise ValueError("QC export readback does not match frozen decisions")
        if len(list(csv.DictReader(io.StringIO(archive.read("decisions.csv").decode("utf-8-sig"))))) != len(snapshot["review"]["decisions"]):
            raise ValueError("QC export lost decisions")
        if "waveform_events" in snapshot["report"]:
            rows = list(csv.DictReader(io.StringIO(archive.read("waveform_events.csv").decode("utf-8-sig"))))
            events = snapshot["report"]["waveform_events"]
            if len(rows) != len(events) or any(row["event_id"] != event["event_id"] or
                                              json.loads(row["stages"]) != event["stages"]
                                              for row, event in zip(rows, events)):
                raise ValueError("QC export lost waveform event stages")
