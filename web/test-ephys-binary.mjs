import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decodeEphysSamples, decodeEphysOverview, prepareEphysSampleRequest } from './ephysBinary.js';

const header = {
    schema_version: 1, recording_id: 'recording-1', artifact_id: 'artifact-1',
    content_sha256: 'a'.repeat(64), reader_version: 'edf-reader/1',
    stream_id: 'stream-1', segment_id: 'segment-2',
    channel_ids: ['channel-4', 'channel-1'], units: ['V', null], shape: [2, 3],
    dtype: '<f8', order: 'C', start_sample: 11, stop_sample: 14,
    sampling_frequency_hz: 500, recording_start_seconds: 17.25,
    clock_id: 'file-clock-1', clock_alignment: 'source_local_unaligned',
    source_sha256: 'b'.repeat(64), source_relative_path: 'source.edf', interval_convention: 'half_open',
    sample_index_origin: 0,
    representation: 'physical', data_kind: 'raw_samples', processing: [],
};
const expected = {
    artifact_id: header.artifact_id, stream_id: header.stream_id,
    segment_id: header.segment_id, channel_ids: [...header.channel_ids],
    start_sample: 11, stop_sample: 14, representation: 'physical',
};
const values = [-25e-6, 0, 50e-6, -3, 0, 7];

function encode(metadata = header, samples = values) {
    const text = new TextEncoder().encode(JSON.stringify(metadata));
    const offset = Math.ceil((8 + text.length) / 8) * 8;
    const bytes = new Uint8Array(offset + samples.length * 8);
    bytes.set([69, 80, 72, 89]);
    new DataView(bytes.buffer).setUint32(4, text.length, true);
    bytes.set(text, 8);
    samples.forEach((value, index) => new DataView(bytes.buffer).setFloat64(offset + index * 8, value, true));
    return bytes;
}

test('channel-major physical values, sign, order, unknown units and segment time survive', () => {
    const decoded = decodeEphysSamples(encode().buffer, expected);
    assert.deepEqual([...decoded.samples], values);
    assert.deepEqual(decoded.metadata, header);
    assert.deepEqual([...decoded.samples.subarray(3, 6)], [-3, 0, 7]);
});

test('unaligned Uint8Array views decode only their own bytes', () => {
    const frame = encode();
    const wrapped = new Uint8Array(frame.length + 5);
    wrapped.set(frame, 3);
    assert.deepEqual([...decodeEphysSamples(wrapped.subarray(3, 3 + frame.length), expected).samples], values);
});

test('digital values have an explicit representation and per-channel units', () => {
    const metadata = { ...header, representation: 'digital', units: ['count', 'count'] };
    const result = decodeEphysSamples(encode(metadata, [-32768, 0, 32767, -1, 2, 3]), { ...expected, representation: 'digital' });
    assert.equal(result.samples[0], -32768);
    assert.deepEqual(result.metadata.units, ['count', 'count']);
});

test('nonfinite raw samples are retained without silent replacement', () => {
    const result = decodeEphysSamples(encode(header, [NaN, Infinity, -Infinity, 0, 1, 2]), expected);
    assert.ok(Number.isNaN(result.samples[0]));
    assert.equal(result.samples[1], Infinity);
});

test('missing magic, truncated header, invalid UTF-8 and invalid JSON reject', () => {
    const magic = encode(); magic[0] = 0;
    assert.throws(() => decodeEphysSamples(magic, expected), /magic/);
    assert.throws(() => decodeEphysSamples(new Uint8Array(7), expected), /truncated/);
    const length = encode(); new DataView(length.buffer).setUint32(4, length.length, true);
    assert.throws(() => decodeEphysSamples(length, expected), /header/);
    const utf8 = encode(); utf8[8] = 255;
    assert.throws(() => decodeEphysSamples(utf8, expected), /header/);
    const json = encode(); json[8] = 120;
    assert.throws(() => decodeEphysSamples(json, expected), /header/);
});

test('inexact payload length and nonzero alignment padding reject', () => {
    const frame = encode();
    assert.throws(() => decodeEphysSamples(frame.subarray(0, frame.length - 1), expected), /length/);
    const extra = new Uint8Array(frame.length + 8); extra.set(frame);
    assert.throws(() => decodeEphysSamples(extra, expected), /length/);
    const metadata = { ...header, reader_version: 'x' };
    while ((8 + new TextEncoder().encode(JSON.stringify(metadata)).length) % 8 === 0) metadata.reader_version += 'x';
    const padded = encode(metadata);
    padded[8 + new DataView(padded.buffer).getUint32(4, true)] = 1;
    assert.throws(() => decodeEphysSamples(padded, expected), /padding/);
});

for (const [field, value] of [
    ['schema_version', 2], ['dtype', '>f8'], ['order', 'F'],
    ['shape', [3, 2]], ['shape', [2, 3.5]], ['shape', [2, Number.MAX_SAFE_INTEGER]],
    ['units', ['V']], ['units', ['V', 0]], ['channel_ids', ['channel-4', 'channel-4']],
    ['sampling_frequency_hz', 0], ['recording_start_seconds', null],
    ['content_sha256', 'unknown'], ['reader_version', ''],
    ['clock_id', ''], ['clock_alignment', 'assumed_aligned'], ['source_sha256', 'unknown'],
    ['source_relative_path', ''], ['interval_convention', 'closed'],
    ['sample_index_origin', -1], ['sample_index_origin', 12],
    ['data_kind', 'overview'], ['processing', ['average_reference']],
]) {
    test(`invalid header ${field}=${JSON.stringify(value)} rejects`, () => {
        assert.throws(() => decodeEphysSamples(encode({ ...header, [field]: value }), expected), { code: 'invalid_ephys_binary' });
    });
}

for (const [field, value] of [
    ['artifact_id', 'artifact-2'], ['stream_id', 'stream-2'], ['segment_id', 'segment-1'],
    ['channel_ids', ['channel-1', 'channel-4']], ['start_sample', 10],
    ['stop_sample', 15], ['representation', 'digital'], ['recording_id', 'recording-2'],
    ['content_sha256', 'b'.repeat(64)], ['reader_version', 'edf-reader/2'],
]) {
    test(`response identity mismatch ${field} rejects`, () => {
        assert.throws(() => decodeEphysSamples(encode(), { ...expected, [field]: value }), { code: 'ephys_identity_mismatch' });
    });
}

test('request serialization preserves channel ordering and separates optional identity constraints', () => {
    const prepared = prepareEphysSampleRequest('artifact/1', { ...expected, recording_id: 'recording-1' });
    assert.deepEqual(prepared.query.getAll('channel_id'), header.channel_ids);
    assert.equal(prepared.query.get('representation'), 'physical');
    assert.equal(prepared.query.has('recording_id'), false);
    assert.equal(prepared.expected.artifact_id, 'artifact/1');
    expected.channel_ids.push('later-mutation');
    assert.equal(prepared.expected.channel_ids.length, 2);
    expected.channel_ids.pop();
});

test('missing/ambiguous request fields cannot trigger guessed units or windows', () => {
    for (const update of [
        { channel_ids: [] }, { channel_ids: ['x', 'x'] }, { representation: undefined },
        { start_sample: -1 }, { start_sample: 1.5 }, { stop_sample: 11 },
        { stop_sample: Number.MAX_SAFE_INTEGER + 1 }, { segment_id: '' },
    ]) assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...expected, ...update }));
    assert.throws(() => decodeEphysSamples(encode(), undefined));
});

const overviewHeader = {
    ...header, data_kind: 'minmax_overview', source_data_kind: 'raw_samples', shape: [2, 2, 2], bin_samples: 2,
    axes: ['channel', 'bin', 'extremum'], extrema: ['min', 'max'], bin_start_samples: [11, 13], bin_stop_samples: [13, 14],
};
const overviewValues = [-25e-6, 0, 50e-6, 50e-6, -3, 0, 7, 7];
const overviewExpected = { ...expected, bin_samples: 2 };

test('overview is explicitly binned, unnormalized and retains the short final bin', () => {
    const result = decodeEphysOverview(encode(overviewHeader, overviewValues), overviewExpected);
    assert.deepEqual([...result.samples], overviewValues);
    assert.deepEqual(result.metadata.bin_stop_samples, [13, 14]);
    assert.throws(() => decodeEphysSamples(encode(overviewHeader, overviewValues), expected), /raw_samples/);
    assert.throws(() => decodeEphysOverview(encode(), overviewExpected), /minmax_overview/);
    assert.equal(prepareEphysSampleRequest('artifact-1', overviewExpected, true).query.get('bin_samples'), '2');
});

const derivedHeader = {
    ...header, data_kind: 'derived_samples', units: ['V', 'V'], sample_index_origin: 10,
    processing: [{ operation: 'explicit_reference', version: 1 }],
    parent_artifact_id: 'raw-artifact-1', parent_content_sha256: 'c'.repeat(64),
    software_versions: { numpy: '2.0.2' },
};
const derivedExpected = { ...expected, data_kind: 'derived_samples' };

test('derived windows require explicit kind, physical volts and complete parent provenance', () => {
    const result = decodeEphysSamples(encode(derivedHeader), derivedExpected);
    assert.deepEqual([...result.samples], values);
    assert.equal(result.metadata.sample_index_origin, 10);
    assert.equal(result.metadata.parent_artifact_id, 'raw-artifact-1');
    assert.throws(() => decodeEphysSamples(encode(derivedHeader), expected), /raw_samples/);
    assert.throws(() => decodeEphysSamples(encode(), derivedExpected), /derived_samples/);
    for (const update of [
        { units: ['V', null] }, { units: ['mV', 'V'] }, { representation: 'digital' },
        { processing: [] }, { parent_artifact_id: '' }, { parent_content_sha256: 'unknown' }, { parent_content_sha256: ['c'.repeat(64)] },
        { software_versions: {} }, { software_versions: { numpy: null } },
    ]) assert.throws(() => decodeEphysSamples(encode({ ...derivedHeader, ...update }), derivedExpected));
    assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...derivedExpected, representation: 'digital' }));
    assert.throws(() => prepareEphysSampleRequest('artifact-1', { ...expected, data_kind: 'filtered_samples' }));
});

test('derived overview retains source kind and cannot masquerade as a raw overview', () => {
    const derivedOverview = { ...derivedHeader, ...overviewHeader, ...derivedHeader,
        data_kind: 'minmax_overview', source_data_kind: 'derived_samples', shape: [2, 2, 2] };
    const result = decodeEphysOverview(encode(derivedOverview, overviewValues), { ...derivedExpected, bin_samples: 2 });
    assert.equal(result.metadata.source_data_kind, 'derived_samples');
    assert.throws(() => decodeEphysOverview(encode(derivedOverview, overviewValues), overviewExpected));
    assert.throws(() => decodeEphysOverview(encode({ ...overviewHeader, source_data_kind: undefined }, overviewValues), overviewExpected));
});

test('overview rejects reshaped axes, wrong bin bounds and changed bin size', () => {
    for (const update of [
        { axes: ['bin', 'channel', 'extremum'] }, { axes: [['channel'], 'bin', 'extremum'] },
        { extrema: ['max', 'min'] }, { extrema: [['min'], 'max'] }, { bin_start_samples: [11, 12] },
        { bin_stop_samples: [13, 15] }, { shape: [2, 4] }, { bin_samples: 0 },
    ]) assert.throws(() => decodeEphysOverview(encode({ ...overviewHeader, ...update }, overviewValues), overviewExpected));
    assert.throws(() => decodeEphysOverview(encode(overviewHeader, overviewValues), { ...overviewExpected, bin_samples: 3 }), { code: 'ephys_identity_mismatch' });
});
