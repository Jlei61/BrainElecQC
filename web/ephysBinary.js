import { ApiError } from './apiError.js';

const MAX_HEADER_BYTES = 4 * 1024 * 1024;
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const IDENTITY_FIELDS = ['artifact_id', 'stream_id', 'segment_id', 'start_sample', 'stop_sample', 'representation'];
const OPTIONAL_IDENTITY_FIELDS = ['recording_id', 'content_sha256', 'reader_version', 'clock_id', 'source_sha256', 'source_relative_path', 'sample_index_origin', 'parent_artifact_id', 'parent_content_sha256'];

function requireContract(condition, message, code = 'invalid_ephys_binary') {
    if (!condition) throw new ApiError({ code, message: `Ephys: ${message}` });
}

function isId(value) {
    return typeof value === 'string' && value.trim().length > 0;
}

function isSample(value) {
    return Number.isSafeInteger(value) && value >= 0;
}

function validateWindow(request, code) {
    for (const key of ['artifact_id', 'stream_id', 'segment_id']) {
        requireContract(isId(request[key]), `missing ${key}`, code);
    }
    requireContract(Array.isArray(request.channel_ids) && request.channel_ids.length > 0
        && request.channel_ids.every(isId)
        && new Set(request.channel_ids).size === request.channel_ids.length, 'invalid channel_ids', code);
    requireContract(isSample(request.start_sample) && isSample(request.stop_sample)
        && request.stop_sample > request.start_sample, 'invalid half-open sample window', code);
    requireContract(['physical', 'digital'].includes(request.representation), 'representation must be explicit', code);
}

/** Serialize a source-sample window without guessing channels, units or bounds. */
export function prepareEphysSampleRequest(artifactId, request, overview = false) {
    requireContract(request && typeof request === 'object', 'missing sample request', 'invalid_ephys_request');
    const expected = { ...request, artifact_id: artifactId, channel_ids: request.channel_ids?.slice(), data_kind: request.data_kind ?? 'raw_samples' };
    validateWindow(expected, 'invalid_ephys_request');
    requireContract(['raw_samples', 'derived_samples'].includes(expected.data_kind), 'unsupported data_kind', 'invalid_ephys_request');
    if (expected.data_kind === 'derived_samples') requireContract(expected.representation === 'physical', 'derived samples require physical representation', 'invalid_ephys_request');
    const query = new URLSearchParams();
    for (const key of IDENTITY_FIELDS.filter(key => key !== 'artifact_id')) query.set(key, String(expected[key]));
    for (const channelId of expected.channel_ids) query.append('channel_id', channelId);
    if (overview) {
        requireContract(Number.isSafeInteger(expected.bin_samples) && expected.bin_samples > 0, 'invalid bin_samples', 'invalid_ephys_request');
        query.set('bin_samples', String(expected.bin_samples));
    }
    return { query, expected };
}

function parseFrame(input) {
    const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
        : input instanceof Uint8Array ? input : null;
    requireContract(bytes && bytes.byteLength >= 8, 'truncated binary frame');
    requireContract(bytes[0] === 69 && bytes[1] === 80 && bytes[2] === 72 && bytes[3] === 89, 'invalid EPHY magic');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const headerLength = view.getUint32(4, true);
    requireContract(headerLength > 0 && headerLength <= MAX_HEADER_BYTES
        && headerLength <= bytes.byteLength - 8, 'invalid header length');
    let metadata;
    try {
        metadata = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(8, 8 + headerLength)));
    } catch {
        requireContract(false, 'invalid UTF-8/JSON header');
    }
    requireContract(metadata && typeof metadata === 'object' && !Array.isArray(metadata), 'invalid header object');
    const offset = Math.ceil((8 + headerLength) / 8) * 8;
    requireContract(offset <= bytes.byteLength, 'truncated alignment padding');
    for (let index = 8 + headerLength; index < offset; index++) requireContract(bytes[index] === 0, 'nonzero alignment padding');
    return { bytes, view, metadata, offset };
}

function validateMetadata(metadata, expected, kind) {
    requireContract(metadata.schema_version === 1, 'unsupported schema_version');
    validateWindow(metadata, 'invalid_ephys_binary');
    for (const key of ['recording_id', 'reader_version', 'clock_id']) requireContract(isId(metadata[key]), `missing ${key}`);
    requireContract(typeof metadata.content_sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.content_sha256), 'invalid content_sha256');
    requireContract(typeof metadata.source_sha256 === 'string' && /^[a-f0-9]{64}$/.test(metadata.source_sha256), 'invalid source_sha256');
    requireContract(isId(metadata.source_relative_path), 'missing source_relative_path');
    requireContract(metadata.interval_convention === 'half_open', 'unsupported interval_convention');
    requireContract(isSample(metadata.sample_index_origin) && metadata.sample_index_origin <= metadata.start_sample, 'invalid sample_index_origin');
    requireContract(metadata.dtype === '<f8' && metadata.order === 'C', 'unsupported dtype/order');
    requireContract(metadata.data_kind === kind, `expected ${kind}`);
    requireContract(Array.isArray(metadata.units) && metadata.units.length === metadata.channel_ids.length
        && metadata.units.every(unit => unit === null || typeof unit === 'string'), 'invalid units');
    if (metadata.representation === 'digital') requireContract(metadata.units.every(unit => unit === 'count'), 'digital units must be count');
    requireContract(Number.isFinite(metadata.sampling_frequency_hz) && metadata.sampling_frequency_hz > 0, 'invalid sampling_frequency_hz');
    requireContract(Number.isFinite(metadata.recording_start_seconds), 'invalid recording_start_seconds');
    requireContract(metadata.clock_alignment === 'source_local_unaligned', 'unsupported clock_alignment');
    requireContract(expected && typeof expected === 'object', 'missing expected identity');
    const sourceKind = expected.data_kind ?? 'raw_samples';
    requireContract(['raw_samples', 'derived_samples'].includes(sourceKind), 'unsupported expected data_kind');
    if (kind === 'minmax_overview') requireContract(metadata.source_data_kind === sourceKind, 'overview source_data_kind differs from request', 'ephys_identity_mismatch');
    requireContract(Array.isArray(metadata.processing), 'missing processing history');
    if (sourceKind === 'raw_samples') requireContract(metadata.processing.length === 0, 'raw-source response contains processing');
    else {
        requireContract(metadata.representation === 'physical' && metadata.units.every(unit => unit === 'V'), 'derived samples require physical volts');
        requireContract(metadata.processing.length > 0, 'derived samples require processing history');
        requireContract(isId(metadata.parent_artifact_id) && typeof metadata.parent_content_sha256 === 'string'
            && /^[a-f0-9]{64}$/.test(metadata.parent_content_sha256), 'invalid derived parent provenance');
        requireContract(metadata.software_versions && typeof metadata.software_versions === 'object'
            && !Array.isArray(metadata.software_versions) && Object.keys(metadata.software_versions).length > 0
            && Object.entries(metadata.software_versions).every(([name, version]) => isId(name) && isId(version)), 'invalid derived software_versions');
    }
    for (const key of IDENTITY_FIELDS) {
        requireContract(metadata[key] === expected[key], `response ${key} differs from request`, 'ephys_identity_mismatch');
    }
    requireContract(Array.isArray(expected.channel_ids) && expected.channel_ids.length === metadata.channel_ids.length
        && metadata.channel_ids.every((id, index) => id === expected.channel_ids[index]), 'response channel order differs from request', 'ephys_identity_mismatch');
    for (const key of OPTIONAL_IDENTITY_FIELDS) {
        if (expected[key] !== undefined) requireContract(metadata[key] === expected[key], `response ${key} differs from catalog`, 'ephys_identity_mismatch');
    }
}

function decodeValues(frame, shape) {
    const { bytes, view, metadata, offset } = frame;
    requireContract(Array.isArray(metadata.shape) && metadata.shape.length === shape.length
        && metadata.shape.every((size, index) => Number.isSafeInteger(size) && size > 0 && size === shape[index]), 'invalid shape');
    const count = shape.reduce((total, size) => total * size, 1);
    requireContract(Number.isSafeInteger(count * 8) && bytes.byteLength - offset === count * 8, 'payload length differs from shape');
    // Network ArrayBuffers are aligned, so the common path does not copy samples.
    // DataView keeps little-endian decoding correct on other hosts/views too.
    let samples;
    if (LITTLE_ENDIAN && (bytes.byteOffset + offset) % 8 === 0) samples = new Float64Array(bytes.buffer, bytes.byteOffset + offset, count);
    else {
        samples = new Float64Array(count);
        for (let index = 0; index < count; index++) samples[index] = view.getFloat64(offset + index * 8, true);
    }
    return { metadata, samples };
}

/**
 * Returns channel-major Float64Array samples plus their exact metadata.
 * recording_start_seconds is window time in clock_id's local source clock;
 * it does not assert alignment between separate files or streams.
 */
export function decodeEphysSamples(input, expected) {
    const frame = parseFrame(input);
    validateMetadata(frame.metadata, expected, expected?.data_kind ?? 'raw_samples');
    return decodeValues(frame, [frame.metadata.channel_ids.length, frame.metadata.stop_sample - frame.metadata.start_sample]);
}

/** Min/max bins with explicit raw/derived source kind, separate from exact samples. */
export function decodeEphysOverview(input, expected) {
    const frame = parseFrame(input);
    const metadata = frame.metadata;
    validateMetadata(metadata, expected, 'minmax_overview');
    requireContract(Number.isSafeInteger(metadata.bin_samples) && metadata.bin_samples > 0, 'invalid bin_samples');
    requireContract(metadata.bin_samples === expected.bin_samples, 'response bin_samples differs from request', 'ephys_identity_mismatch');
    requireContract(Array.isArray(metadata.axes) && metadata.axes.length === 3
        && metadata.axes.every((axis, index) => axis === ['channel', 'bin', 'extremum'][index]), 'invalid overview axes');
    requireContract(Array.isArray(metadata.extrema) && metadata.extrema.length === 2
        && metadata.extrema.every((extremum, index) => extremum === ['min', 'max'][index]), 'invalid overview extrema');
    const bins = Math.ceil((metadata.stop_sample - metadata.start_sample) / metadata.bin_samples);
    for (const key of ['bin_start_samples', 'bin_stop_samples']) requireContract(Array.isArray(metadata[key]) && metadata[key].length === bins, `invalid ${key}`);
    for (let index = 0; index < bins; index++) {
        const start = metadata.start_sample + index * metadata.bin_samples;
        requireContract(metadata.bin_start_samples[index] === start
            && metadata.bin_stop_samples[index] === Math.min(start + metadata.bin_samples, metadata.stop_sample), 'overview bin sample bounds mismatch');
    }
    return decodeValues(frame, [metadata.channel_ids.length, bins, 2]);
}
