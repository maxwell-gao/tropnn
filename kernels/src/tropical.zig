const std = @import("std");
const parallel = @import("parallel.zig");
const simd = @import("simd.zig");

inline fn bestCellF32Generic(
    latent: [*]const f32,
    router_weight: [*]const f32,
    router_bias: [*]const f32,
    head: usize,
    cells: usize,
    code_dim: usize,
) usize {
    var best_cell: usize = 0;
    var best_score: f32 = -std.math.inf(f32);
    const head_cell_base = head * cells;

    var cell: usize = 0;
    while (cell < cells) : (cell += 1) {
        const offset = (head_cell_base + cell) * code_dim;
        const score = simd.dot(latent, router_weight + offset, code_dim) + router_bias[head_cell_base + cell];
        if (score > best_score) {
            best_score = score;
            best_cell = cell;
        }
    }
    return best_cell;
}

inline fn bestCellF32Small(
    comptime cell_count: usize,
    latent: [*]const f32,
    router_weight: [*]const f32,
    router_bias: [*]const f32,
    head: usize,
    code_dim: usize,
) usize {
    const head_cell_base = head * cell_count;
    const weight_base = head_cell_base * code_dim;
    const zero: simd.VecF32 = @splat(0.0);
    var acc = [_]simd.VecF32{zero} ** cell_count;

    var i: usize = 0;
    while (i + simd.VEC_SIZE <= code_dim) : (i += simd.VEC_SIZE) {
        const x = simd.load(latent + i);
        inline for (0..cell_count) |cell| {
            acc[cell] += x * simd.load(router_weight + weight_base + cell * code_dim + i);
        }
    }

    var scores: [cell_count]f32 = undefined;
    inline for (0..cell_count) |cell| {
        scores[cell] = @reduce(.Add, acc[cell]) + router_bias[head_cell_base + cell];
    }
    while (i < code_dim) : (i += 1) {
        const x = latent[i];
        inline for (0..cell_count) |cell| {
            scores[cell] += x * router_weight[weight_base + cell * code_dim + i];
        }
    }

    var best_cell: usize = 0;
    var best_score: f32 = scores[0];
    inline for (1..cell_count) |cell| {
        if (scores[cell] > best_score) {
            best_score = scores[cell];
            best_cell = cell;
        }
    }
    return best_cell;
}

inline fn bestCellF32(
    latent: [*]const f32,
    router_weight: [*]const f32,
    router_bias: [*]const f32,
    head: usize,
    cells: usize,
    code_dim: usize,
) usize {
    return switch (cells) {
        2 => bestCellF32Small(2, latent, router_weight, router_bias, head, code_dim),
        3 => bestCellF32Small(3, latent, router_weight, router_bias, head, code_dim),
        4 => bestCellF32Small(4, latent, router_weight, router_bias, head, code_dim),
        else => bestCellF32Generic(latent, router_weight, router_bias, head, cells, code_dim),
    };
}

inline fn bestCellF16Generic(
    latent: [*]const f32,
    router_weight: [*]const f16,
    router_bias: [*]const f16,
    head: usize,
    cells: usize,
    code_dim: usize,
) usize {
    var best_cell: usize = 0;
    var best_score: f32 = -std.math.inf(f32);
    const head_cell_base = head * cells;

    var cell: usize = 0;
    while (cell < cells) : (cell += 1) {
        const offset = (head_cell_base + cell) * code_dim;
        const bias = @as(f32, @floatCast(router_bias[head_cell_base + cell]));
        const score = simd.dotF16(latent, router_weight + offset, code_dim) + bias;
        if (score > best_score) {
            best_score = score;
            best_cell = cell;
        }
    }
    return best_cell;
}

inline fn bestCellF16Small(
    comptime cell_count: usize,
    latent: [*]const f32,
    router_weight: [*]const f16,
    router_bias: [*]const f16,
    head: usize,
    code_dim: usize,
) usize {
    const head_cell_base = head * cell_count;
    const weight_base = head_cell_base * code_dim;
    const zero: simd.VecF32 = @splat(0.0);
    var acc = [_]simd.VecF32{zero} ** cell_count;

    var i: usize = 0;
    while (i + simd.VEC_SIZE <= code_dim) : (i += simd.VEC_SIZE) {
        const x = simd.load(latent + i);
        inline for (0..cell_count) |cell| {
            acc[cell] += x * simd.loadF16ToF32(router_weight + weight_base + cell * code_dim + i);
        }
    }

    var scores: [cell_count]f32 = undefined;
    inline for (0..cell_count) |cell| {
        scores[cell] = @reduce(.Add, acc[cell]) + @as(f32, @floatCast(router_bias[head_cell_base + cell]));
    }
    while (i < code_dim) : (i += 1) {
        const x = latent[i];
        inline for (0..cell_count) |cell| {
            scores[cell] += x * @as(f32, @floatCast(router_weight[weight_base + cell * code_dim + i]));
        }
    }

    var best_cell: usize = 0;
    var best_score: f32 = scores[0];
    inline for (1..cell_count) |cell| {
        if (scores[cell] > best_score) {
            best_score = scores[cell];
            best_cell = cell;
        }
    }
    return best_cell;
}

inline fn bestCellF16(
    latent: [*]const f32,
    router_weight: [*]const f16,
    router_bias: [*]const f16,
    head: usize,
    cells: usize,
    code_dim: usize,
) usize {
    return switch (cells) {
        2 => bestCellF16Small(2, latent, router_weight, router_bias, head, code_dim),
        3 => bestCellF16Small(3, latent, router_weight, router_bias, head, code_dim),
        4 => bestCellF16Small(4, latent, router_weight, router_bias, head, code_dim),
        else => bestCellF16Generic(latent, router_weight, router_bias, head, cells, code_dim),
    };
}

const TropRouteF32Context = struct {
    heads: usize,
    cells: usize,
    code_dim: usize,
    code_scale: f32,
    latent: [*]const f32,
    router_weight: [*]const f32,
    router_bias: [*]const f32,
    code: [*]const f32,
    hidden: [*]f32,
};

inline fn tropRouteF32RowGeneric(ctx: *TropRouteF32Context, latent_ptr: [*]const f32, hidden_ptr: [*]f32) void {
    var head: usize = 0;
    while (head < ctx.heads) : (head += 1) {
        const cell = bestCellF32Generic(latent_ptr, ctx.router_weight, ctx.router_bias, head, ctx.cells, ctx.code_dim);
        const code_base = (head * ctx.cells + cell) * ctx.code_dim;
        simd.addScaled(hidden_ptr, ctx.code + code_base, ctx.code_scale, ctx.code_dim);
    }
}

inline fn tropRouteF32RowSmall(comptime cell_count: usize, ctx: *TropRouteF32Context, latent_ptr: [*]const f32, hidden_ptr: [*]f32) void {
    const head_stride = cell_count * ctx.code_dim;
    var head: usize = 0;
    while (head < ctx.heads) : (head += 1) {
        const cell = bestCellF32Small(cell_count, latent_ptr, ctx.router_weight, ctx.router_bias, head, ctx.code_dim);
        const code_base = head * head_stride + cell * ctx.code_dim;
        simd.addScaled(hidden_ptr, ctx.code + code_base, ctx.code_scale, ctx.code_dim);
    }
}

fn tropRouteF32RangeGeneric(ctx: *TropRouteF32Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const latent_ptr = ctx.latent + row * ctx.code_dim;
        const hidden_ptr = ctx.hidden + row * ctx.code_dim;
        simd.copy(hidden_ptr, latent_ptr, ctx.code_dim);
        tropRouteF32RowGeneric(ctx, latent_ptr, hidden_ptr);
    }
}

fn tropRouteF32RangeSmall(comptime cell_count: usize, ctx: *TropRouteF32Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const latent_ptr = ctx.latent + row * ctx.code_dim;
        const hidden_ptr = ctx.hidden + row * ctx.code_dim;
        simd.copy(hidden_ptr, latent_ptr, ctx.code_dim);
        tropRouteF32RowSmall(cell_count, ctx, latent_ptr, hidden_ptr);
    }
}

fn tropRouteF32RangeDispatched(ctx: *TropRouteF32Context, row_start: usize, row_end: usize) void {
    switch (ctx.cells) {
        2 => tropRouteF32RangeSmall(2, ctx, row_start, row_end),
        3 => tropRouteF32RangeSmall(3, ctx, row_start, row_end),
        4 => tropRouteF32RangeSmall(4, ctx, row_start, row_end),
        else => tropRouteF32RangeGeneric(ctx, row_start, row_end),
    }
}

const TropRouteF16Context = struct {
    heads: usize,
    cells: usize,
    code_dim: usize,
    code_scale: f32,
    latent: [*]const f32,
    router_weight: [*]const f16,
    router_bias: [*]const f16,
    code: [*]const f16,
    hidden: [*]f32,
};

inline fn tropRouteF16RowGeneric(ctx: *TropRouteF16Context, latent_ptr: [*]const f32, hidden_ptr: [*]f32) void {
    var head: usize = 0;
    while (head < ctx.heads) : (head += 1) {
        const cell = bestCellF16Generic(latent_ptr, ctx.router_weight, ctx.router_bias, head, ctx.cells, ctx.code_dim);
        const code_base = (head * ctx.cells + cell) * ctx.code_dim;
        simd.addScaledF16(hidden_ptr, ctx.code + code_base, ctx.code_scale, ctx.code_dim);
    }
}

inline fn tropRouteF16RowSmall(comptime cell_count: usize, ctx: *TropRouteF16Context, latent_ptr: [*]const f32, hidden_ptr: [*]f32) void {
    const head_stride = cell_count * ctx.code_dim;
    var head: usize = 0;
    while (head < ctx.heads) : (head += 1) {
        const cell = bestCellF16Small(cell_count, latent_ptr, ctx.router_weight, ctx.router_bias, head, ctx.code_dim);
        const code_base = head * head_stride + cell * ctx.code_dim;
        simd.addScaledF16(hidden_ptr, ctx.code + code_base, ctx.code_scale, ctx.code_dim);
    }
}

fn tropRouteF16RangeGeneric(ctx: *TropRouteF16Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const latent_ptr = ctx.latent + row * ctx.code_dim;
        const hidden_ptr = ctx.hidden + row * ctx.code_dim;
        simd.copy(hidden_ptr, latent_ptr, ctx.code_dim);
        tropRouteF16RowGeneric(ctx, latent_ptr, hidden_ptr);
    }
}

fn tropRouteF16RangeSmall(comptime cell_count: usize, ctx: *TropRouteF16Context, row_start: usize, row_end: usize) void {
    var row = row_start;
    while (row < row_end) : (row += 1) {
        const latent_ptr = ctx.latent + row * ctx.code_dim;
        const hidden_ptr = ctx.hidden + row * ctx.code_dim;
        simd.copy(hidden_ptr, latent_ptr, ctx.code_dim);
        tropRouteF16RowSmall(cell_count, ctx, latent_ptr, hidden_ptr);
    }
}

fn tropRouteF16RangeDispatched(ctx: *TropRouteF16Context, row_start: usize, row_end: usize) void {
    switch (ctx.cells) {
        2 => tropRouteF16RangeSmall(2, ctx, row_start, row_end),
        3 => tropRouteF16RangeSmall(3, ctx, row_start, row_end),
        4 => tropRouteF16RangeSmall(4, ctx, row_start, row_end),
        else => tropRouteF16RangeGeneric(ctx, row_start, row_end),
    }
}

/// Batch TropLinear route/code forward with f32 router and code parameters.
/// Shape contract:
/// - latent: [item_count, code_dim]
/// - router_weight: [heads, cells, code_dim]
/// - router_bias: [heads, cells]
/// - code: [heads, cells, code_dim]
/// - hidden: [item_count, code_dim], overwritten by this kernel
export fn trop_route_hidden_batch_f32(
    item_count: usize,
    heads: usize,
    cells: usize,
    code_dim: usize,
    code_scale: f32,
    latent: [*]const f32,
    router_weight: [*]const f32,
    router_bias: [*]const f32,
    code: [*]const f32,
    hidden: [*]f32,
) void {
    var ctx = TropRouteF32Context{
        .heads = heads,
        .cells = cells,
        .code_dim = code_dim,
        .code_scale = code_scale,
        .latent = latent,
        .router_weight = router_weight,
        .router_bias = router_bias,
        .code = code,
        .hidden = hidden,
    };
    parallel.parallelFor(TropRouteF32Context, &ctx, item_count, 128, tropRouteF32RangeDispatched);
}

/// Batch TropLinear route/code forward with f16 router/code parameters and f32 accumulation.
export fn trop_route_hidden_batch_f16(
    item_count: usize,
    heads: usize,
    cells: usize,
    code_dim: usize,
    code_scale: f32,
    latent: [*]const f32,
    router_weight: [*]const f16,
    router_bias: [*]const f16,
    code: [*]const f16,
    hidden: [*]f32,
) void {
    var ctx = TropRouteF16Context{
        .heads = heads,
        .cells = cells,
        .code_dim = code_dim,
        .code_scale = code_scale,
        .latent = latent,
        .router_weight = router_weight,
        .router_bias = router_bias,
        .code = code,
        .hidden = hidden,
    };
    parallel.parallelFor(TropRouteF16Context, &ctx, item_count, 128, tropRouteF16RangeDispatched);
}

test "f32 tropical route selects best cells" {
    const item_count: usize = 2;
    const heads: usize = 2;
    const cells: usize = 2;
    const code_dim: usize = 2;
    const latent = [_]f32{ 2.0, 1.0, -1.0, 3.0 };
    const router_weight = [_]f32{
        1.0,  0.0,
        0.0,  1.0,
        -1.0, 0.0,
        0.0,  -1.0,
    };
    const router_bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const code = [_]f32{
        10.0, 0.0,
        0.0,  20.0,
        30.0, 0.0,
        0.0,  40.0,
    };
    var hidden = [_]f32{0.0} ** (item_count * code_dim);

    trop_route_hidden_batch_f32(item_count, heads, cells, code_dim, 0.5, &latent, &router_weight, &router_bias, &code, &hidden);

    try std.testing.expectApproxEqAbs(@as(f32, 7.0), hidden[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), hidden[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), hidden[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), hidden[3], 1e-6);
}
