const std = @import("std");
const simd = @import("simd.zig");

inline fn computeComparisonsWithOffsets(
    input: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
    num_comparisons: usize,
) usize {
    return switch (num_comparisons) {
        1 => computeComparisonsStatic(1, input, anchors, offsets, table_idx),
        2 => computeComparisonsStatic(2, input, anchors, offsets, table_idx),
        3 => computeComparisonsStatic(3, input, anchors, offsets, table_idx),
        4 => computeComparisonsStatic(4, input, anchors, offsets, table_idx),
        5 => computeComparisonsStatic(5, input, anchors, offsets, table_idx),
        6 => computeComparisonsStatic(6, input, anchors, offsets, table_idx),
        7 => computeComparisonsStatic(7, input, anchors, offsets, table_idx),
        8 => computeComparisonsStatic(8, input, anchors, offsets, table_idx),
        else => computeComparisonsGeneric(input, anchors, offsets, table_idx, num_comparisons),
    };
}

inline fn computeComparisonsStatic(
    comptime comparison_count: usize,
    input: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
) usize {
    var idx: usize = 0;
    const anchor_base = table_idx * comparison_count * 2;
    const offset_base = table_idx * comparison_count;

    inline for (0..comparison_count) |r| {
        const a = anchors[anchor_base + r * 2];
        const b = anchors[anchor_base + r * 2 + 1];
        const threshold = offsets[offset_base + r];
        const margin = input[a] - input[b] - threshold;
        idx |= @as(usize, @intFromBool(margin > 0.0)) << @intCast(r);
    }
    return idx;
}

inline fn computeComparisonsGeneric(
    input: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
    num_comparisons: usize,
) usize {
    var idx: usize = 0;
    const anchor_base = table_idx * num_comparisons * 2;
    const offset_base = table_idx * num_comparisons;

    var r: usize = 0;
    while (r < num_comparisons) : (r += 1) {
        const a = anchors[anchor_base + r * 2];
        const b = anchors[anchor_base + r * 2 + 1];
        const threshold = offsets[offset_base + r];
        const margin = input[a] - input[b] - threshold;
        idx |= @as(usize, @intFromBool(margin > 0.0)) << @intCast(r);
    }
    return idx;
}

/// Batch pairwise-LUT forward with f32 weights.
/// Shape contract:
/// - weights: [num_tables, 2^num_comparisons, output_dim]
/// - anchors: [num_tables, num_comparisons, 2], usize/int64 ABI on x86_64
/// - offsets: [num_tables, num_comparisons]
/// - inputs: [batch_size, input_dim]
/// - outputs: [batch_size, output_dim], zeroed by this kernel
export fn lut_forward_batch_with_offsets_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    weights: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);

    var row: usize = 0;
    while (row < batch_size) : (row += 1) {
        const input_ptr = inputs + row * input_dim;
        const output_ptr = outputs + row * output_dim;
        simd.zero(output_ptr, output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx, num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 1, num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 2, num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 3, num_comparisons);
            const stride = table_size * output_dim;
            const base0 = table_idx * stride + idx0 * output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * output_dim;

            if (table_idx + simd.PREFETCH_L2 < num_tables) {
                @prefetch(&anchors[(table_idx + simd.PREFETCH_L2) * num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&offsets[(table_idx + simd.PREFETCH_L2) * num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4Sources(output_ptr, weights + base0, weights + base1, weights + base2, weights + base3, output_dim);
        }
        while (table_idx < num_tables) : (table_idx += 1) {
            const idx = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx, num_comparisons);
            const weight_base = table_idx * table_size * output_dim + idx * output_dim;
            simd.accumulate4x(output_ptr, weights + weight_base, output_dim);
        }
    }
}

/// Batch pairwise-LUT forward with f16 weights and f32 accumulation.
export fn lut_forward_batch_f16_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    weights_f16: [*]const f16,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);

    var row: usize = 0;
    while (row < batch_size) : (row += 1) {
        const input_ptr = inputs + row * input_dim;
        const output_ptr = outputs + row * output_dim;
        simd.zero(output_ptr, output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx, num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 1, num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 2, num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 3, num_comparisons);
            const stride = table_size * output_dim;
            const base0 = table_idx * stride + idx0 * output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * output_dim;

            if (table_idx + simd.PREFETCH_L2 < num_tables) {
                @prefetch(&anchors[(table_idx + simd.PREFETCH_L2) * num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&offsets[(table_idx + simd.PREFETCH_L2) * num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4SourcesF16(output_ptr, weights_f16 + base0, weights_f16 + base1, weights_f16 + base2, weights_f16 + base3, output_dim);
        }
        while (table_idx < num_tables) : (table_idx += 1) {
            const idx = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx, num_comparisons);
            const weight_base = table_idx * table_size * output_dim + idx * output_dim;
            simd.accumulate4xF16(output_ptr, weights_f16 + weight_base, output_dim);
        }
    }
}

test "comparison hash uses thresholds" {
    const x = [_]f32{ 3.0, 1.0, -1.0 };
    const anchors = [_]usize{ 0, 1, 1, 2 };
    const offsets = [_]f32{ 1.5, 3.0 };
    try std.testing.expectEqual(@as(usize, 1), computeComparisonsWithOffsets(&x, &anchors, &offsets, 0, 2));
}
