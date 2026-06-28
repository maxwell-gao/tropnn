const std = @import("std");
const parallel = @import("parallel.zig");
const simd = @import("simd.zig");

const MAX_PAGED_CODES: usize = 256;
const MAX_PAGED_ROWS: usize = 4096;
const MAX_TREE_TABLES: usize = 256;
const TREE_DIM_TILE: usize = 64;

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

inline fn computeComparisonsSoAWithOffsets(
    input: [*]const f32,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
    num_comparisons: usize,
) usize {
    return switch (num_comparisons) {
        1 => computeComparisonsSoAStatic(1, input, anchor_a, anchor_b, offsets, table_idx),
        2 => computeComparisonsSoAStatic(2, input, anchor_a, anchor_b, offsets, table_idx),
        3 => computeComparisonsSoAStatic(3, input, anchor_a, anchor_b, offsets, table_idx),
        4 => computeComparisonsSoAStatic(4, input, anchor_a, anchor_b, offsets, table_idx),
        5 => computeComparisonsSoAStatic(5, input, anchor_a, anchor_b, offsets, table_idx),
        6 => computeComparisonsSoAStatic(6, input, anchor_a, anchor_b, offsets, table_idx),
        7 => computeComparisonsSoAStatic(7, input, anchor_a, anchor_b, offsets, table_idx),
        8 => computeComparisonsSoAStatic(8, input, anchor_a, anchor_b, offsets, table_idx),
        else => computeComparisonsSoAGeneric(input, anchor_a, anchor_b, offsets, table_idx, num_comparisons),
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

inline fn computeComparisonsSoAStatic(
    comptime comparison_count: usize,
    input: [*]const f32,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
) usize {
    var idx: usize = 0;
    const base = table_idx * comparison_count;

    inline for (0..comparison_count) |r| {
        const a = anchor_a[base + r];
        const b = anchor_b[base + r];
        const threshold = offsets[base + r];
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

inline fn computeComparisonsSoAGeneric(
    input: [*]const f32,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    table_idx: usize,
    num_comparisons: usize,
) usize {
    var idx: usize = 0;
    const base = table_idx * num_comparisons;

    var r: usize = 0;
    while (r < num_comparisons) : (r += 1) {
        const a = anchor_a[base + r];
        const b = anchor_b[base + r];
        const threshold = offsets[base + r];
        const margin = input[a] - input[b] - threshold;
        idx |= @as(usize, @intFromBool(margin > 0.0)) << @intCast(r);
    }
    return idx;
}

const LutForwardF32Context = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    weights: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

const LutForwardF32SoAContext = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    weights: [*]const f32,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

const LutForwardF32PagedContext = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    page_size: usize,
    weights: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

const LutForwardF32TreeTiledContext = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    weights: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

fn lutForwardF32RowMajorRange(ctx: *LutForwardF32Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const input_ptr = ctx.inputs + row * ctx.input_dim;
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const stride = ctx.table_size * ctx.output_dim;
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
                @prefetch(&ctx.anchors[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4Sources(output_ptr, ctx.weights + base0, ctx.weights + base1, ctx.weights + base2, ctx.weights + base3, ctx.output_dim);
        }
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4x(output_ptr, ctx.weights + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF32SoARowMajorRange(ctx: *LutForwardF32SoAContext, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const input_ptr = ctx.inputs + row * ctx.input_dim;
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const stride = ctx.table_size * ctx.output_dim;
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
                @prefetch(&ctx.anchor_a[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.anchor_b[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4Sources(output_ptr, ctx.weights + base0, ctx.weights + base1, ctx.weights + base2, ctx.weights + base3, ctx.output_dim);
        }
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            const idx = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4x(output_ptr, ctx.weights + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF32TableMajorRange(ctx: *LutForwardF32Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);
    }

    const stride = ctx.table_size * ctx.output_dim;
    var table_idx: usize = 0;
    while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
        if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
            @prefetch(&ctx.anchors[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
            @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
        }

        row = row_start;
        while (row < row_end) : (row += 1) {
            const input_ptr = ctx.inputs + row * ctx.input_dim;
            const output_ptr = ctx.outputs + row * ctx.output_dim;
            const idx0 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            simd.accumulate4Sources(output_ptr, ctx.weights + base0, ctx.weights + base1, ctx.weights + base2, ctx.weights + base3, ctx.output_dim);
        }
    }
    while (table_idx < ctx.num_tables) : (table_idx += 1) {
        row = row_start;
        while (row < row_end) : (row += 1) {
            const input_ptr = ctx.inputs + row * ctx.input_dim;
            const output_ptr = ctx.outputs + row * ctx.output_dim;
            const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4x(output_ptr, ctx.weights + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF32TreeTiledRange(ctx: *LutForwardF32TreeTiledContext, row_start: usize, row_end: usize) void {
    var codes: [MAX_TREE_TABLES]usize = undefined;
    var acc: [TREE_DIM_TILE]f32 = undefined;
    const stride = ctx.table_size * ctx.output_dim;

    var row = row_start;
    while (row < row_end) : (row += 1) {
        const input_ptr = ctx.inputs + row * ctx.input_dim;
        const output_ptr = ctx.outputs + row * ctx.output_dim;

        var table_idx: usize = 0;
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            codes[table_idx] = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
        }

        var d0: usize = 0;
        while (d0 < ctx.output_dim) : (d0 += TREE_DIM_TILE) {
            const width = @min(TREE_DIM_TILE, ctx.output_dim - d0);
            var d: usize = 0;
            while (d < width) : (d += 1) {
                acc[d] = 0.0;
            }

            table_idx = 0;
            while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
                const base0 = table_idx * stride + codes[table_idx] * ctx.output_dim + d0;
                const base1 = (table_idx + 1) * stride + codes[table_idx + 1] * ctx.output_dim + d0;
                const base2 = (table_idx + 2) * stride + codes[table_idx + 2] * ctx.output_dim + d0;
                const base3 = (table_idx + 3) * stride + codes[table_idx + 3] * ctx.output_dim + d0;
                if (table_idx + 8 < ctx.num_tables) {
                    const pf = (table_idx + 8) * stride + codes[table_idx + 8] * ctx.output_dim + d0;
                    @prefetch(ctx.weights + pf, .{ .rw = .read, .locality = 2, .cache = .data });
                }
                d = 0;
                while (d < width) : (d += 1) {
                    acc[d] += ctx.weights[base0 + d] + ctx.weights[base1 + d] + ctx.weights[base2 + d] + ctx.weights[base3 + d];
                }
            }
            while (table_idx < ctx.num_tables) : (table_idx += 1) {
                const base = table_idx * stride + codes[table_idx] * ctx.output_dim + d0;
                d = 0;
                while (d < width) : (d += 1) {
                    acc[d] += ctx.weights[base + d];
                }
            }

            d = 0;
            while (d < width) : (d += 1) {
                output_ptr[d0 + d] = acc[d];
            }
        }
    }
}

fn lutForwardF32PagedRange(ctx: *LutForwardF32PagedContext, row_start: usize, row_end: usize) void {
    var counts: [MAX_PAGED_CODES]usize = undefined;
    var offsets: [MAX_PAGED_CODES + 1]usize = undefined;
    var cursor: [MAX_PAGED_CODES]usize = undefined;
    var rows_by_code: [MAX_PAGED_ROWS]usize = undefined;

    var page_start = row_start;
    while (page_start < row_end) {
        const page_end = @min(row_end, page_start + ctx.page_size);
        var row = page_start;
        while (row < page_end) : (row += 1) {
            simd.zero(ctx.outputs + row * ctx.output_dim, ctx.output_dim);
        }

        const stride = ctx.table_size * ctx.output_dim;
        var table_idx: usize = 0;
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            var code: usize = 0;
            while (code < ctx.table_size) : (code += 1) {
                counts[code] = 0;
            }

            row = page_start;
            while (row < page_end) : (row += 1) {
                const input_ptr = ctx.inputs + row * ctx.input_dim;
                const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
                counts[idx] += 1;
            }

            offsets[0] = 0;
            code = 0;
            while (code < ctx.table_size) : (code += 1) {
                offsets[code + 1] = offsets[code] + counts[code];
                cursor[code] = offsets[code];
            }

            row = page_start;
            while (row < page_end) : (row += 1) {
                const input_ptr = ctx.inputs + row * ctx.input_dim;
                const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
                const slot = cursor[idx];
                rows_by_code[slot] = row;
                cursor[idx] = slot + 1;
            }

            code = 0;
            while (code < ctx.table_size) : (code += 1) {
                const begin = offsets[code];
                const end = offsets[code + 1];
                if (begin == end) continue;
                const weight_ptr = ctx.weights + table_idx * stride + code * ctx.output_dim;
                var pos = begin;
                while (pos < end) : (pos += 1) {
                    const out_row = rows_by_code[pos];
                    simd.accumulate4x(ctx.outputs + out_row * ctx.output_dim, weight_ptr, ctx.output_dim);
                }
            }
        }

        page_start = page_end;
    }
}

const LutForwardF16Context = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    weights_f16: [*]const f16,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

const LutForwardF16SoAContext = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    weights_f16: [*]const f16,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

const LutForwardF16PagedContext = struct {
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    table_size: usize,
    page_size: usize,
    weights_f16: [*]const f16,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
};

fn lutForwardF16RowMajorRange(ctx: *LutForwardF16Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const input_ptr = ctx.inputs + row * ctx.input_dim;
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const stride = ctx.table_size * ctx.output_dim;
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
                @prefetch(&ctx.anchors[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4SourcesF16(output_ptr, ctx.weights_f16 + base0, ctx.weights_f16 + base1, ctx.weights_f16 + base2, ctx.weights_f16 + base3, ctx.output_dim);
        }
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4xF16(output_ptr, ctx.weights_f16 + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF16SoARowMajorRange(ctx: *LutForwardF16SoAContext, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const input_ptr = ctx.inputs + row * ctx.input_dim;
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);

        var table_idx: usize = 0;
        while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
            const idx0 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const stride = ctx.table_size * ctx.output_dim;
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
                @prefetch(&ctx.anchor_a[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.anchor_b[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
            }

            simd.accumulate4SourcesF16(output_ptr, ctx.weights_f16 + base0, ctx.weights_f16 + base1, ctx.weights_f16 + base2, ctx.weights_f16 + base3, ctx.output_dim);
        }
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            const idx = computeComparisonsSoAWithOffsets(input_ptr, ctx.anchor_a, ctx.anchor_b, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4xF16(output_ptr, ctx.weights_f16 + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF16TableMajorRange(ctx: *LutForwardF16Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const output_ptr = ctx.outputs + row * ctx.output_dim;
        simd.zero(output_ptr, ctx.output_dim);
    }

    const stride = ctx.table_size * ctx.output_dim;
    var table_idx: usize = 0;
    while (table_idx + 4 <= ctx.num_tables) : (table_idx += 4) {
        if (table_idx + simd.PREFETCH_L2 < ctx.num_tables) {
            @prefetch(&ctx.anchors[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons * 2], .{ .rw = .read, .locality = 3, .cache = .data });
            @prefetch(&ctx.offsets[(table_idx + simd.PREFETCH_L2) * ctx.num_comparisons], .{ .rw = .read, .locality = 3, .cache = .data });
        }

        row = row_start;
        while (row < row_end) : (row += 1) {
            const input_ptr = ctx.inputs + row * ctx.input_dim;
            const output_ptr = ctx.outputs + row * ctx.output_dim;
            const idx0 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const idx1 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 1, ctx.num_comparisons);
            const idx2 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 2, ctx.num_comparisons);
            const idx3 = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx + 3, ctx.num_comparisons);
            const base0 = table_idx * stride + idx0 * ctx.output_dim;
            const base1 = (table_idx + 1) * stride + idx1 * ctx.output_dim;
            const base2 = (table_idx + 2) * stride + idx2 * ctx.output_dim;
            const base3 = (table_idx + 3) * stride + idx3 * ctx.output_dim;

            simd.accumulate4SourcesF16(output_ptr, ctx.weights_f16 + base0, ctx.weights_f16 + base1, ctx.weights_f16 + base2, ctx.weights_f16 + base3, ctx.output_dim);
        }
    }
    while (table_idx < ctx.num_tables) : (table_idx += 1) {
        row = row_start;
        while (row < row_end) : (row += 1) {
            const input_ptr = ctx.inputs + row * ctx.input_dim;
            const output_ptr = ctx.outputs + row * ctx.output_dim;
            const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
            const weight_base = table_idx * ctx.table_size * ctx.output_dim + idx * ctx.output_dim;
            simd.accumulate4xF16(output_ptr, ctx.weights_f16 + weight_base, ctx.output_dim);
        }
    }
}

fn lutForwardF16PagedRange(ctx: *LutForwardF16PagedContext, row_start: usize, row_end: usize) void {
    var counts: [MAX_PAGED_CODES]usize = undefined;
    var offsets: [MAX_PAGED_CODES + 1]usize = undefined;
    var cursor: [MAX_PAGED_CODES]usize = undefined;
    var rows_by_code: [MAX_PAGED_ROWS]usize = undefined;

    var page_start = row_start;
    while (page_start < row_end) {
        const page_end = @min(row_end, page_start + ctx.page_size);
        var row = page_start;
        while (row < page_end) : (row += 1) {
            simd.zero(ctx.outputs + row * ctx.output_dim, ctx.output_dim);
        }

        const stride = ctx.table_size * ctx.output_dim;
        var table_idx: usize = 0;
        while (table_idx < ctx.num_tables) : (table_idx += 1) {
            var code: usize = 0;
            while (code < ctx.table_size) : (code += 1) {
                counts[code] = 0;
            }

            row = page_start;
            while (row < page_end) : (row += 1) {
                const input_ptr = ctx.inputs + row * ctx.input_dim;
                const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
                counts[idx] += 1;
            }

            offsets[0] = 0;
            code = 0;
            while (code < ctx.table_size) : (code += 1) {
                offsets[code + 1] = offsets[code] + counts[code];
                cursor[code] = offsets[code];
            }

            row = page_start;
            while (row < page_end) : (row += 1) {
                const input_ptr = ctx.inputs + row * ctx.input_dim;
                const idx = computeComparisonsWithOffsets(input_ptr, ctx.anchors, ctx.offsets, table_idx, ctx.num_comparisons);
                const slot = cursor[idx];
                rows_by_code[slot] = row;
                cursor[idx] = slot + 1;
            }

            code = 0;
            while (code < ctx.table_size) : (code += 1) {
                const begin = offsets[code];
                const end = offsets[code + 1];
                if (begin == end) continue;
                const weight_ptr = ctx.weights_f16 + table_idx * stride + code * ctx.output_dim;
                var pos = begin;
                while (pos < end) : (pos += 1) {
                    const out_row = rows_by_code[pos];
                    simd.accumulate4xF16(ctx.outputs + out_row * ctx.output_dim, weight_ptr, ctx.output_dim);
                }
            }
        }

        page_start = page_end;
    }
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
    var ctx = LutForwardF32Context{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .weights = weights,
        .anchors = anchors,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    if (parallel.tropnn_get_num_threads() <= 1) {
        parallel.parallelFor(LutForwardF32Context, &ctx, batch_size, 64, lutForwardF32TableMajorRange);
    } else {
        parallel.parallelFor(LutForwardF32Context, &ctx, batch_size, 64, lutForwardF32RowMajorRange);
    }
}

/// Batch pairwise-LUT forward with f32 weights, table-level code interleaving, and dim-tiled accumulation.
export fn lut_forward_batch_tree_tiled_with_offsets_no_cache(
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
    if (num_tables > MAX_TREE_TABLES) {
        lut_forward_batch_with_offsets_no_cache(batch_size, num_tables, num_comparisons, input_dim, output_dim, weights, anchors, offsets, inputs, outputs);
        return;
    }
    const table_size = @as(usize, 1) << @intCast(num_comparisons);
    var ctx = LutForwardF32TreeTiledContext{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .weights = weights,
        .anchors = anchors,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    parallel.parallelFor(LutForwardF32TreeTiledContext, &ctx, batch_size, 64, lutForwardF32TreeTiledRange);
}

/// Batch pairwise-LUT forward with f32 weights and structure-of-arrays anchor layout.
/// Shape contract:
/// - anchor_a: [num_tables, num_comparisons]
/// - anchor_b: [num_tables, num_comparisons]
export fn lut_forward_batch_soa_with_offsets_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    weights: [*]const f32,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);
    var ctx = LutForwardF32SoAContext{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .weights = weights,
        .anchor_a = anchor_a,
        .anchor_b = anchor_b,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    parallel.parallelFor(LutForwardF32SoAContext, &ctx, batch_size, 64, lutForwardF32SoARowMajorRange);
}

/// Page-style pairwise-LUT forward with f32 weights.
/// For each row tile and table, rows are grouped by route code before payload rows are applied.
export fn lut_forward_batch_paged_with_offsets_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    page_size: usize,
    weights: [*]const f32,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);
    if (table_size > MAX_PAGED_CODES) {
        lut_forward_batch_with_offsets_no_cache(batch_size, num_tables, num_comparisons, input_dim, output_dim, weights, anchors, offsets, inputs, outputs);
        return;
    }
    const effective_page_size = @max(@as(usize, 1), @min(page_size, MAX_PAGED_ROWS));
    var ctx = LutForwardF32PagedContext{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .page_size = effective_page_size,
        .weights = weights,
        .anchors = anchors,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    parallel.parallelFor(LutForwardF32PagedContext, &ctx, batch_size, effective_page_size, lutForwardF32PagedRange);
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
    var ctx = LutForwardF16Context{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .weights_f16 = weights_f16,
        .anchors = anchors,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    if (parallel.tropnn_get_num_threads() <= 1) {
        parallel.parallelFor(LutForwardF16Context, &ctx, batch_size, 64, lutForwardF16TableMajorRange);
    } else {
        parallel.parallelFor(LutForwardF16Context, &ctx, batch_size, 64, lutForwardF16RowMajorRange);
    }
}

/// Batch pairwise-LUT forward with f16 weights, f32 accumulation, and structure-of-arrays anchor layout.
export fn lut_forward_batch_soa_f16_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    weights_f16: [*]const f16,
    anchor_a: [*]const usize,
    anchor_b: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);
    var ctx = LutForwardF16SoAContext{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .weights_f16 = weights_f16,
        .anchor_a = anchor_a,
        .anchor_b = anchor_b,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    parallel.parallelFor(LutForwardF16SoAContext, &ctx, batch_size, 64, lutForwardF16SoARowMajorRange);
}

/// Page-style pairwise-LUT forward with f16 weights and f32 accumulation.
export fn lut_forward_batch_paged_f16_no_cache(
    batch_size: usize,
    num_tables: usize,
    num_comparisons: usize,
    input_dim: usize,
    output_dim: usize,
    page_size: usize,
    weights_f16: [*]const f16,
    anchors: [*]const usize,
    offsets: [*]const f32,
    inputs: [*]const f32,
    outputs: [*]f32,
) void {
    const table_size = @as(usize, 1) << @intCast(num_comparisons);
    if (table_size > MAX_PAGED_CODES) {
        lut_forward_batch_f16_no_cache(batch_size, num_tables, num_comparisons, input_dim, output_dim, weights_f16, anchors, offsets, inputs, outputs);
        return;
    }
    const effective_page_size = @max(@as(usize, 1), @min(page_size, MAX_PAGED_ROWS));
    var ctx = LutForwardF16PagedContext{
        .num_tables = num_tables,
        .num_comparisons = num_comparisons,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .table_size = table_size,
        .page_size = effective_page_size,
        .weights_f16 = weights_f16,
        .anchors = anchors,
        .offsets = offsets,
        .inputs = inputs,
        .outputs = outputs,
    };
    parallel.parallelFor(LutForwardF16PagedContext, &ctx, batch_size, effective_page_size, lutForwardF16PagedRange);
}

test "comparison hash uses thresholds" {
    const x = [_]f32{ 3.0, 1.0, -1.0 };
    const anchors = [_]usize{ 0, 1, 1, 2 };
    const offsets = [_]f32{ 1.5, 3.0 };
    try std.testing.expectEqual(@as(usize, 1), computeComparisonsWithOffsets(&x, &anchors, &offsets, 0, 2));
}
