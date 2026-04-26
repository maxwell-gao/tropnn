const std = @import("std");
const simd = @import("simd.zig");

inline fn computeComparisonsWithOffsets(
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
    @memset(outputs[0 .. batch_size * output_dim], 0.0);

    var table_idx: usize = 0;
    while (table_idx < num_tables) : (table_idx += 1) {
        if (table_idx + simd.PREFETCH_L2 < num_tables) {
            @prefetch(&anchors[(table_idx + simd.PREFETCH_L2) * num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
            @prefetch(&offsets[(table_idx + simd.PREFETCH_L2) * num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
        }

        var row: usize = 0;
        while (row < batch_size) : (row += 1) {
            const input_ptr = inputs + row * input_dim;
            const output_ptr = outputs + row * output_dim;
            const idx = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx, num_comparisons);
            const weight_base = (table_idx * table_size + idx) * output_dim;
            simd.accumulate(output_ptr, weights + weight_base, output_dim);
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
    @memset(outputs[0 .. batch_size * output_dim], 0.0);

    var row: usize = 0;
    while (row < batch_size) : (row += 1) {
        const input_ptr = inputs + row * input_dim;
        const output_ptr = outputs + row * output_dim;

        if (num_tables == 0) continue;
        var cur_idx = computeComparisonsWithOffsets(input_ptr, anchors, offsets, 0, num_comparisons);

        var table_idx: usize = 0;
        while (table_idx < num_tables) : (table_idx += 1) {
            const idx = cur_idx;
            const weight_base = table_idx * table_size * output_dim + idx * output_dim;

            if (table_idx + 1 < num_tables) {
                cur_idx = computeComparisonsWithOffsets(input_ptr, anchors, offsets, table_idx + 1, num_comparisons);
                const next_base = (table_idx + 1) * table_size * output_dim + cur_idx * output_dim;
                @prefetch(&weights_f16[next_base], .{ .rw = .read, .locality = 2, .cache = .data });
            }

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
