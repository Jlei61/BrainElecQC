"""Technical header inspection against original EDF/BDF bytes, without patient text."""
from datetime import datetime
import hashlib
import math
from pathlib import Path

from ..storage import hash_file


VERSION = "eeg-header-qc-1.0"
CHECKS = [("signature", "格式签名", 1), ("layout", "头长度与通道数", 1),
          ("channels", "通道名称与顺序", 1), ("sampling", "逐通道采样率", 2),
          ("calibration", "单位与幅值标定", 2), ("payload", "样本数与文件长度", 2),
          ("clock", "记录开始时间与分段", 1)]


def _number(value, integer=False):
    try:
        number = float(value.decode("ascii").strip())
        if not math.isfinite(number) or (integer and not number.is_integer()):
            return None
        return int(number) if integer else number
    except (UnicodeDecodeError, ValueError):
        return None


def inspect_input_header(path, *, expected_sha256=None):
    """Return scored EDF/BDF header checks, or an explicit unsupported-format status.

    Identity fields are recorded as presence flags only. Sampling rates remain
    per channel; anonymous patient fields and optional filter text do not deduct.
    """
    path = Path(path)
    before = path.stat()
    fingerprint = hash_file(path)["sha256"]
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("Header source fingerprint differs from the recording; select its original input")
    with path.open("rb") as handle:
        fixed = handle.read(256)
        format_name = "BDF" if fixed[:8] == b"\xffBIOSEMI" else "EDF" if fixed[:8] == b"0       " else None
        result = {"schema_version": VERSION, "source_sha256": fingerprint, "format": format_name,
                  "file_size_bytes": before.st_size, "checks": [], "channels": [], "notes": []}
        if format_name is None and path.suffix.lower() not in {".edf", ".bdf"}:
            return {**result, "status": "format_confirmation_required", "auto_points": None,
                    "reason": "待按设备与文件版本核对头字段、采样流、分段索引和伴随文件"}
        checks = {}
        def check(key, ok, detail):
            checks[key] = (bool(ok), detail)
        check("signature", format_name is not None, "EDF 16 位" if format_name == "EDF" else
              "BDF 24 位" if format_name == "BDF" else "文件签名缺失或不匹配")
        ns, size = _number(fixed[252:256], True), _number(fixed[184:192], True)
        layout = len(fixed) == 256 and ns is not None and 1 <= ns <= 4096 and size == 256*(ns+1)
        raw = handle.read(size-256) if layout else b""
        layout = layout and len(raw) == size-256
        check("layout", layout, f"{ns} 道，头长度 {size} 字节" if layout else "头长度、通道数或逐通道字段不完整")
        duration, records = _number(fixed[244:252]), _number(fixed[236:244], True)
        numeric = []
        if layout:
            offset, fields = 0, [("raw_name", 16), ("transducer", 80), ("unit", 8),
                ("physical_min", 8), ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
                ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]
            channels = [{"source_index": i} for i in range(ns)]
            for field, width in fields:
                for i, channel in enumerate(channels):
                    value = raw[offset+i*width:offset+(i+1)*width]
                    channel[field] = (_number(value, field in {"digital_min", "digital_max", "samples_per_record"})
                        if field in {"physical_min", "physical_max", "digital_min", "digital_max", "samples_per_record"}
                        else value.decode("latin1").strip())
                offset += width*ns
            numeric = [c for c in channels if c["raw_name"] not in {"EDF Annotations", "BDF Annotations"}]
            check("channels", bool(numeric) and all(c["raw_name"] for c in numeric),
                  f"{len(numeric)} 个信号通道，{ns-len(numeric)} 个注释通道；按源索引保留顺序")
            sampling = bool(numeric) and duration is not None and duration > 0 and all(
                c["samples_per_record"] is not None and c["samples_per_record"] > 0 for c in channels)
            rates = sorted({c["samples_per_record"]/duration for c in numeric}) if sampling else []
            sampling = sampling and all(math.isfinite(r) for r in rates)
            if not sampling:
                rates = []
            check("sampling", sampling, " / ".join(f"{r:g} Hz" for r in rates) if sampling else "每记录样本数或时长缺失/无效")
            limit = 2**(23 if format_name == "BDF" else 15)
            invalid = []
            for c in numeric:
                is_biosemi_status = format_name == "BDF" and c["source_index"] == ns-1 and c["raw_name"].lower() == "status"
                valid = (bool(c["unit"]) or is_biosemi_status) and all(c[k] is not None for k in
                    ("physical_min", "physical_max", "digital_min", "digital_max"))
                valid = valid and -limit <= c["digital_min"] < c["digital_max"] < limit and c["physical_min"] != c["physical_max"]
                if valid:
                    gain = (c["physical_max"]-c["physical_min"])/(c["digital_max"]-c["digital_min"])
                    valid = math.isfinite(gain) and math.isfinite(c["physical_min"]-gain*c["digital_min"])
                if not valid:
                    invalid.append(c["source_index"])
                result["channels"].append({k: c[k] for k in ("source_index", "raw_name", "unit", "physical_min",
                    "physical_max", "digital_min", "digital_max", "samples_per_record")})
                result["channels"][-1]["sampling_rate_hz"] = c["samples_per_record"]/duration if sampling else None
            check("calibration", bool(numeric) and not invalid,
                  "逐通道单位及数字量→物理量标定完整" if not invalid else f"标定缺项/无效，源通道索引 {invalid}")
            width = 3 if format_name == "BDF" else 2
            record_bytes = width*sum(c["samples_per_record"] for c in channels) if sampling else 0
            payload = before.st_size-size
            actual = payload//record_bytes if record_bytes else None
            complete = record_bytes > 0 and payload > 0 and payload % record_bytes == 0 and (records == -1 or records == actual)
            check("payload", complete, f"声明 {records} 条，实际 {actual} 条；每条 {duration} s" if record_bytes else "数据记录长度待核")
            if records == -1 and complete:
                result["notes"].append("记录数按完整数据区精确推导")
            result.update(record_count=actual, record_duration_seconds=duration, sampling_rates_hz=rates,
                          header_sha256=hashlib.sha256(fixed+raw).hexdigest())
            result["notes"].append(f"采集滤波说明已填写 {sum(bool(c['prefilter']) for c in numeric)}/{len(numeric)} 道")
        else:
            for key in ("channels", "sampling", "calibration", "payload"):
                check(key, False, "头结构不完整，需修复后核对")
        try:
            datetime.strptime(fixed[168:184].decode("ascii"), "%d.%m.%y%H.%M.%S")
            clock_ok = True
        except (ValueError, UnicodeDecodeError):
            clock_ok = False
        clock_detail = "开始日期/时间格式有效；相对时间来自数据记录"
        if layout and checks["payload"][0] and checks["calibration"][0] and checks["sampling"][0] and format_name:
            from ..readers.edf import EdfBdfReader
            try:
                description = EdfBdfReader(path, "header-check").describe()
                warnings = description["warnings"]
                issues = [w["code"] for w in warnings if w["code"] not in {"record_count_from_complete_payload", "unknown_physical_quantity"}]
                if issues:
                    clock_ok = False
                    clock_detail = "时间/注释需核对："+", ".join(issues)
            except ValueError as exc:
                clock_ok, clock_detail = False, "分段时间解析需核对："+str(exc)
        check("clock", clock_ok, clock_detail if clock_ok or clock_detail.startswith(("时间", "分段")) else "开始日期/时间字段缺失或无效")
        result["identity_fields"] = {"patient_field_present": bool(fixed[8:88].strip()),
                                     "recording_field_present": bool(fixed[88:168].strip())}
        for key, label, weight in CHECKS:
            passed, detail = checks[key]
            result["checks"].append({"id": key, "label": label, "max_points": weight,
                "points": weight if passed else 0, "status": "pass" if passed else "needs_review", "detail": detail})
        result["auto_points"] = sum(c["points"] for c in result["checks"])
        result["status"] = "pass" if result["auto_points"] == 10 else "needs_review"
        result["reason"] = "技术头字段完整，数据长度与声明一致" if result["status"] == "pass" else "；".join(
            c["label"]+"："+c["detail"] for c in result["checks"] if c["status"] != "pass")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Input changed during header inspection")
    return result


def score_header(block):
    if block is None:
        return {"score": None, "reason": "待读取原始头文件", "metrics": {}, "evidence": []}
    return {"score": block["auto_points"], "reason": block["reason"], "metrics": block,
            "components": [{"id": c["id"], "deduction": c["max_points"]-c["points"]} for c in block["checks"]], "evidence": []}


def render_header_checks(block):
    if not block:
        return []
    from .report import _text
    lines = ["## 头文件检查", "", "| 检查内容 | 结果 | 核对信息 |", "| --- | --- | --- |"]
    for c in block["checks"]:
        lines.append(f"| {c['label']} | {'通过' if c['status'] == 'pass' else '需核对'} | {_text(c['detail'])} |")
    return lines+["", "采集参考、硬件滤波及事件说明随源文件保存；身份字段按匿名化状态记录。", ""]
